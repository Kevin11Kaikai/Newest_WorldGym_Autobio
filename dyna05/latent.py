"""Fine-tune the ORIGINAL latent transition module on real simulator transitions.

The private bridge must expose its actual torch module as documented. This file
does not guess the Stage 4 architecture or train on archived imagined states.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import Observation, confusion, write_json


def dtype_of(module):
    return next(module.parameters()).dtype


def tensor(value, module):
    return torch.as_tensor(value, dtype=dtype_of(module), device='cpu')


def validate_module(module):
    if not isinstance(module, torch.nn.Module):
        raise TypeError('Bridge trainable_model must be the original torch.nn.Module wrapper')
    for method in ('encode', 'transition', 'decode'):
        if not callable(getattr(module, method, None)):
            raise TypeError(f'Original model wrapper needs {method}()')
    names = dict(module.named_parameters())
    dynamics = set(getattr(module, 'dynamics_parameter_names', []))
    readout = set(getattr(module, 'readout_parameter_names', []))
    if not dynamics or not (dynamics | readout) <= names.keys() or dynamics & readout:
        raise ValueError('Explicit, disjoint transition/readout parameter names required')
    for field in ('state_mean', 'state_scale'):
        if not isinstance(getattr(module, field, None), torch.Tensor):
            raise ValueError('Wrapper must export original frozen state normalization buffers')
    if (module.state_mean.ndim != 1 or module.state_scale.shape != module.state_mean.shape
            or (module.state_scale <= 0).any()
            or not torch.isfinite(module.state_mean).all() or not torch.isfinite(module.state_scale).all()):
        raise ValueError('Invalid original state normalization')
    if any(p.device.type != 'cpu' for p in module.parameters()):
        raise ValueError('Minimal reproducible trainer currently requires CPU tensors')
    return dynamics, readout


class TorchLatentEnv:
    def __init__(self, module):
        validate_module(module)
        self.module = module.eval()

    @torch.no_grad()
    def reset(self, observation):
        observation.validate('world_model')
        self.z = self.module.encode(tensor(observation.policy_state[None], self.module))
        return copy.deepcopy(observation)

    @torch.no_grad()
    def step(self, packet):
        self.z = self.module.transition(self.z, tensor(np.asarray(packet)[None], self.module))
        decoded = self.module.decode(self.z)
        state = decoded['policy_state'][0].detach().cpu().numpy()
        obs = Observation(state, float(decoded['height'].reshape(-1)[0]),
                          float(decoded['contact_logits'].reshape(-1)[0].sigmoid()))
        obs.validate('world_model')
        return obs


def check_equivalence(original_env, module, initial, packets):
    """Before optimization, compare recurrent predictions with the frozen bridge."""
    env = TorchLatentEnv(module)
    obs = copy.deepcopy(initial.observation)
    obs.official_success = None
    original_env.reset(copy.deepcopy(obs))
    env.reset(copy.deepcopy(obs))
    errors = []
    for packet in packets:
        a, b = original_env.step(packet), env.step(packet)
        va, vb = np.r_[a.policy_state, a.height, a.contact], np.r_[b.policy_state, b.height, b.contact]
        if va.shape != vb.shape or not np.allclose(va, vb, atol=1e-6, rtol=1e-5):
            raise ValueError('Trainable wrapper does not reproduce original recurrent model predictions')
        errors.append(float(np.max(np.abs(va - vb))))
    return dict(packets=len(errors), maximum_absolute_error=max(errors))


def check_episodes(episodes, split, evidence_kind):
    if not episodes:
        raise ValueError('No physical episodes available for dynamics fitting/validation')
    for ep in episodes:
        if ep.source != 'simulator' or ep.split != split or ep.evidence_kind != evidence_kind:
            raise ValueError('Dynamics data must come from the designated simulator split')
        n = len(ep.actions)
        if n == 0 or len(ep.states) != n + 1 or ep.terminated[:-1].any():
            raise ValueError('Misaligned or cross-terminal simulator transitions')
        if not all(np.isfinite(x).all() for x in (ep.states, ep.actions, ep.height, ep.contact)):
            raise ValueError('Non-finite simulator training data')


def window_loss(module, episode, start, horizon):
    """Recursive predictions; true future states appear only on the target side."""
    states = tensor(episode.states, module)
    z = module.encode(states[start:start + 1]).detach()
    losses = []
    for t in range(start, min(start + horizon, len(episode.actions))):
        z = module.transition(z, tensor(episode.actions[t:t + 1], module))
        with torch.no_grad():
            target_z = module.encode(states[t + 1:t + 2])
        prediction = module.decode(z)
        state_loss = ((prediction['policy_state'] - states[t + 1:t + 2]) / module.state_scale).square().mean()
        latent_loss = F.mse_loss(z, target_z)
        height_loss = ((prediction['height'].reshape(-1) - tensor([episode.height[t + 1]], module)) / .05).square().mean()
        contact_loss = F.binary_cross_entropy_with_logits(prediction['contact_logits'].reshape(-1),
                                                         tensor([episode.contact[t + 1]], module))
        losses.append(state_loss + latent_loss + height_loss + contact_loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_dynamics(module, episodes, evidence_kind):
    check_episodes(episodes, 'wm_validate', evidence_kind)
    module.eval()
    one_step, rollout_state, height, contact, actual, predicted = [], [], [], [], [], []
    for ep in episodes:
        one_step.append(float(torch.stack([window_loss(module, ep, t, 1) for t in range(len(ep.actions))]).mean()))
        z = module.encode(tensor(ep.states[:1], module))
        hits = []
        for t, action in enumerate(ep.actions):
            z = module.transition(z, tensor(action[None], module))
            output = module.decode(z)
            error = (output['policy_state'] - tensor(ep.states[t + 1:t + 2], module)) / module.state_scale
            rollout_state.append(float(error.square().mean()))
            h = float(output['height'].reshape(-1)[0])
            c = float(output['contact_logits'].reshape(-1)[0].sigmoid())
            height.append((h - ep.height[t + 1]) ** 2)
            contact.append((c - ep.contact[t + 1]) ** 2)
            hits.append(h > .925 and c >= .5)
        actual.append(ep.success)
        predicted.append(any(hits))
    metrics = dict(one_step_loss=float(np.mean(one_step)),
                   open_loop_state_nmse=float(np.mean(rollout_state)),
                   open_loop_height_mse=float(np.mean(height)),
                   open_loop_contact_brier=float(np.mean(contact)),
                   reward_errors=confusion(actual, predicted))
    if not np.isfinite([v for v in metrics.values() if isinstance(v, float)]).all():
        raise FloatingPointError('Non-finite model validation')
    return metrics


def fit_dynamics(original, train, validation, output, steps=40, horizon=3, lr=1e-4,
                 seed=110607, evidence_kind='mujoco'):
    """Single fixed-budget fit; select once using held-out validation, never final test."""
    check_episodes(train, 'wm_fit', evidence_kind)
    check_episodes(validation, 'wm_validate', evidence_kind)
    if {e.seed for e in train} & {e.seed for e in validation}:
        raise ValueError('Dynamics train/validation initial states overlap')
    if steps < 1 or horizon < 1:
        raise ValueError('Nonempty training schedule required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    module = copy.deepcopy(original)
    dynamics, readout = validate_module(module)
    module.eval()  # preserve frozen encoder/normalizer and disable dropout/BN updates
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name in dynamics | readout)
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    before = evaluate_dynamics(original, validation, evidence_kind)
    losses = []
    for _ in range(steps):
        ep = train[int(torch.randint(len(train), (), generator=generator))]
        start = int(torch.randint(len(ep.actions), (), generator=generator))
        loss = window_loss(module, ep, start, horizon)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite latent training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Non-finite latent parameters')
        losses.append(float(loss.detach()))
    original_state, trained_state = original.state_dict(), module.state_dict()
    changed = [name for name in dynamics if not torch.equal(original_state[name], trained_state[name])]
    if not changed:
        raise RuntimeError('No latent transition parameter changed; reject head-only/no-op fit')
    # Frozen tensors, including encoder buffers, must remain byte-identical.
    for name in original_state.keys() - (dynamics | readout):
        if not torch.equal(original_state[name], trained_state[name]):
            raise RuntimeError(f'Unexpected update outside dynamics/readout: {name}')
    after = evaluate_dynamics(module, validation, evidence_kind)
    accepted = (after['one_step_loss'] < before['one_step_loss']
                and after['open_loop_state_nmse'] <= before['open_loop_state_nmse']
                and after['open_loop_height_mse'] <= before['open_loop_height_mse']
                and after['open_loop_contact_brier'] <= before['open_loop_contact_brier']
                and after['reward_errors']['fp'] <= before['reward_errors']['fp']
                and after['reward_errors']['fn'] <= before['reward_errors']['fn'])
    report = dict(before=before, after=after, accepted=bool(accepted), optimizer_steps=steps,
                  changed_dynamics_parameters=sorted(changed), training_losses=losses,
                  evidence_kind=evidence_kind, validation_kind='open_loop_on_simulator_actions')
    torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), steps=steps, horizon=horizon, lr=lr,
                    dynamics_parameters=sorted(dynamics), readout_parameters=sorted(readout)), output / 'candidate.pt')
    torch.save((module if accepted else original).state_dict(), output / 'selected.pt')
    write_json(output / 'validation.json', report)
    return module if accepted else copy.deepcopy(original), report
