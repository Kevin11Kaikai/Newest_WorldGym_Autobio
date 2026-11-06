"""Packet-level GRPO for a stochastic residual on the frozen Stage 4 controller.

One stochastic decision is one 10 x 7 packet, not seventy independent tokens.
No simulator reward is manufactured here: imagined rewards are explicitly labelled.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .core import rollout, write_json


def group_advantages(rewards, eps=1e-8):
    """Population standard deviation; constant groups contribute zero gradient."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError('Expected finite [initial states, samples >= 2] rewards')
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    advantage = (rewards - rewards.mean(dim=1, keepdim=True)) / std.clamp_min(eps)
    return torch.where(std > eps, advantage, torch.zeros_like(advantage)), std.squeeze(1) <= eps


def clipped_objective(logp, old_logp, advantage, mask, clip=.2):
    """Mean within each episode, then mean across episodes (padding excluded)."""
    if mask.ndim != 2 or (mask.sum(1) == 0).any():
        raise ValueError('Every episode needs at least one active packet')
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage[:, None],
                              ratio.clamp(1 - clip, 1 + clip) * advantage[:, None])
    return ((surrogate * mask).sum(1) / mask.sum(1)).mean()


class ResidualPolicy(nn.Module):
    """Fixed-variance Gaussian -> tanh -> feasible physical residual interval.

    A residual is shared by the ten commands. Intersecting their feasible intervals
    prevents post-sampling clipping. Zero-width axes are deterministic and masked.
    The seventh (gripper) actuator remains exactly the frozen controller's command.
    """
    def __init__(self, state_mean, state_scale, bounds, radius=.01, std=.5, hidden=32):
        super().__init__()
        mean = torch.as_tensor(state_mean, dtype=torch.float64)
        scale = torch.as_tensor(state_scale, dtype=torch.float64)
        limits = torch.as_tensor(bounds, dtype=torch.float64)
        if (mean.ndim != 1 or scale.shape != mean.shape or (scale <= 0).any()
                or limits.shape != (7, 2) or (limits[:, 1] <= limits[:, 0]).any()
                or not torch.isfinite(torch.cat([mean, scale, limits.flatten()])).all()
                or not 0 < radius <= .05 or not 0 < std <= 2):
            raise ValueError('Invalid frozen normalization, bounds or residual scale')
        self.register_buffer('state_mean', mean)
        self.register_buffer('state_scale', scale)
        self.register_buffer('bounds', limits)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float64))
        self.radius, self.hidden = radius, hidden
        self.net = nn.Sequential(nn.Linear(len(mean), hidden), nn.Tanh(), nn.Linear(hidden, 6)).double()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net((state - self.state_mean) / self.state_scale)

    def interval(self, base):
        if base.shape[-2:] != (10, 7) or not torch.isfinite(base).all():
            raise ValueError('Expected finite physical 10 x 7 packet')
        # Baseline violations must be repaired in the original controller, not hidden.
        if ((base < self.bounds[:, 0]) | (base > self.bounds[:, 1])).any():
            raise ValueError('Frozen controller emits commands outside actuator bounds')
        radius = self.radius * (self.bounds[:6, 1] - self.bounds[:6, 0])
        low = torch.maximum((self.bounds[:6, 0] - base[..., :6]).amax(-2), -radius)
        high = torch.minimum((self.bounds[:6, 1] - base[..., :6]).amin(-2), radius)
        half = (high - low) / 2
        # Numerically collapsed axes must also be deterministic when EXECUTED,
        # not merely omitted from the density. Keep the feasible midpoint.
        half = torch.where(half > 1e-12, half, 0.)
        return (low + high) / 2, half

    def log_prob(self, state, u, half):
        mean = self(state)
        normal = -.5 * ((u - mean) / self.std).square() - self.std.log() - .5 * math.log(2 * math.pi)
        # Stable log(1 - tanh(u)^2), including saturated samples.
        jacobian = 2 * (math.log(2) - u - F.softplus(-2 * u))
        active = half > 1e-12
        return torch.where(active, normal - half.clamp_min(1e-12).log() - jacobian, 0.).sum(-1)

    def reference_kl(self, state, half, reference):
        # KL is invariant under the common bijective action transform.
        current, ref = self(state), reference(state).detach()
        kl = (reference.std / self.std).log() + (self.std.square() + (current - ref).square()) / (2 * reference.std.square()) - .5
        return torch.where(half > 1e-12, kl, 0.).sum(-1)

    @torch.no_grad()
    def sample(self, state, base, generator):
        state = torch.as_tensor(state, dtype=torch.float64)
        base = torch.as_tensor(base, dtype=torch.float64)
        center, half = self.interval(base)
        u = self(state) + self.std * torch.randn(6, generator=generator, dtype=torch.float64)
        residual = center + half * u.tanh()
        packet = base.clone()
        packet[:, :6] += residual
        if ((packet < self.bounds[:, 0]) | (packet > self.bounds[:, 1])).any():
            raise FloatingPointError('Numerical action-bound violation; no silent clipping')
        return packet.numpy(), dict(state=state, u=u, half=half,
                                    old_logp=self.log_prob(state, u, half), base=base)

    def spec(self):
        return dict(state_mean=self.state_mean.tolist(), state_scale=self.state_scale.tolist(),
                    bounds=self.bounds.tolist(), radius=self.radius, std=float(self.std), hidden=self.hidden)


class ResidualController:
    def __init__(self, baseline, policy, generator):
        self.baseline, self.policy, self.generator = baseline, policy, generator
        self.decisions = []

    def reset(self):
        self.baseline.reset()
        self.decisions = []

    def packet(self, obs):
        packet, decision = self.policy.sample(obs.policy_state, self.baseline.packet(obs), self.generator)
        self.decisions.append(decision)
        return packet


class ExactPacket:
    """Use the generic recorder with identity application; no hidden action clipping."""
    def __init__(self, name):
        self.name = name

    def apply(self, packet, bounds):
        packet = np.asarray(packet)
        if packet.shape != (10, 7) or not np.isfinite(packet).all() or ((packet < bounds[:, 0]) | (packet > bounds[:, 1])).any():
            raise ValueError('Invalid physical action packet')
        return packet, 0


def policy_episode(env, baseline, policy, initial, bounds, horizon, source, split, evidence, rng_seed, name):
    generator = torch.Generator().manual_seed(int(rng_seed))
    controller = baseline if policy is None else ResidualController(baseline, policy, generator)
    ep = rollout(env, controller, initial, ExactPacket(name), bounds, horizon, source, split, evidence)
    return ep, [] if policy is None else controller.decisions


def pack_decisions(episodes):
    if not episodes or any(not episode for episode in episodes):
        raise ValueError('Empty GRPO batch/episode')
    n, horizon = len(episodes), max(map(len, episodes))
    if not n or not horizon:
        raise ValueError('Empty GRPO batch')
    result = {}
    for key in ('state', 'u', 'half', 'old_logp'):
        result[key] = torch.zeros((n, horizon) + episodes[0][0][key].shape, dtype=torch.float64)
    result['mask'] = torch.zeros(n, horizon, dtype=torch.float64)
    for i, episode in enumerate(episodes):
        for t, decision in enumerate(episode):
            for key in ('state', 'u', 'half', 'old_logp'):
                result[key][i, t] = decision[key]
            result['mask'][i, t] = 1
    return result


class GRPOTrainer:
    def __init__(self, policy, reference=None, lr=3e-4, beta=.01, clip=.2, epochs=2, seed=110606):
        self.policy = policy
        self.reference = copy.deepcopy(policy if reference is None else reference).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.generator = torch.Generator().manual_seed(seed)
        self.beta, self.clip, self.epochs = beta, clip, epochs
        self.iteration, self.optimizer_steps = 0, 0

    def update(self, decisions, rewards):
        advantages, constant = group_advantages(rewards)
        batch = pack_decisions(decisions)
        advantage = advantages.flatten()
        if len(advantage) != len(decisions):
            raise ValueError('Group/trajectory count mismatch')
        before = torch.cat([p.detach().flatten().clone() for p in self.policy.parameters()])
        losses = []
        grad_norm = 0.
        # No arbitrary shaping or KL-only optimization for a fully constant batch.
        if not constant.all():
            for _ in range(self.epochs):
                logp = self.policy.log_prob(batch['state'], batch['u'], batch['half'])
                objective = clipped_objective(logp, batch['old_logp'], advantage, batch['mask'], self.clip)
                kl = self.policy.reference_kl(batch['state'], batch['half'], self.reference)
                kl = ((kl * batch['mask']).sum(1) / batch['mask'].sum(1)).mean()
                loss = -objective + self.beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite GRPO objective')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 1., error_if_nonfinite=True)
                grad_norm = float(norm)
                self.optimizer.step()
                if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
                    raise FloatingPointError('Non-finite policy parameters')
                self.optimizer_steps += 1
                losses.append(float(loss.detach()))
        self.iteration += 1
        after = torch.cat([p.detach().flatten() for p in self.policy.parameters()])
        return dict(iteration=self.iteration, optimizer_steps=self.optimizer_steps,
                    reward_source='world_model', mean_imagined_reward=float(rewards.mean()),
                    constant_groups=int(constant.sum()), groups=len(constant), losses=losses,
                    grad_norm=grad_norm, parameter_delta_l2=float((after - before).norm()))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(policy_spec=self.policy.spec(), policy=self.policy.state_dict(),
                        reference=self.reference.state_dict(), optimizer=self.optimizer.state_dict(),
                        generator=self.generator.get_state(), torch_rng=torch.get_rng_state(),
                        iteration=self.iteration, optimizer_steps=self.optimizer_steps,
                        beta=self.beta, clip=self.clip, epochs=self.epochs), path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        policy = ResidualPolicy(**saved['policy_spec'])
        policy.load_state_dict(saved['policy'])
        obj = cls(policy, beta=saved['beta'], clip=saved['clip'], epochs=saved['epochs'])
        obj.reference.load_state_dict(saved['reference'])
        obj.optimizer.load_state_dict(saved['optimizer'])
        obj.generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        obj.iteration, obj.optimizer_steps = saved['iteration'], saved['optimizer_steps']
        return obj


def train_imagined(trainer, model, baseline, initials, bounds, horizon, groupsize, iterations, output):
    """Group members share initial state; rollout RNG is independent and checkpointed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logs = []
    for _ in range(iterations):
        decisions, rewards = [], []
        iteration = trainer.iteration
        for initial in initials:
            group = []
            for member in range(groupsize):
                seed = int(torch.randint(0, 2**31 - 1, (), generator=trainer.generator))
                ep, trace = policy_episode(model, baseline, trainer.policy, initial, bounds, horizon,
                                          'world_model', 'grpo_train', 'imagined', seed, 'stochastic_residual')
                stem = output / f'iter_{iteration:03d}_state_{initial.seed}_member_{member}'
                ep.save(stem.with_suffix('.npz'))
                torch.save(dict(decisions=trace, rng_seed=seed, reward=float(ep.success),
                                reward_source='world_model', initial_hash=initial.digest), stem.with_suffix('.pt'))
                decisions.append(trace)
                group.append(float(ep.success))
            rewards.append(group)
        logs.append(trainer.update(decisions, torch.tensor(rewards, dtype=torch.float64)))
        trainer.save(output / 'checkpoint.pt')
        write_json(output / f'update_{iteration:03d}.json', logs[-1])
    return logs
