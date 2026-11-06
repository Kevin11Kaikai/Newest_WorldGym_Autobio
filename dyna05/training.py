"""One bounded Dyna round: GRPO -> physical data -> latent fit -> GRPO -> physical test."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import numpy as np
import torch
from .core import write_json, confusion, paired_difference, check_bridge, ranking_metrics
from .grpo import ResidualPolicy, GRPOTrainer, policy_episode, train_imagined
from .latent import TorchLatentEnv, fit_dynamics, validate_module, check_equivalence


@dataclass
class TrainingConfig:
    horizon_packets: int = 112
    group_size: int = 4
    grpo_iterations: int = 3
    grpo_epochs: int = 2
    policy_lr: float = 3e-4
    kl_beta: float = .01
    residual_radius: float = .01
    residual_std: float = .5
    dynamics_steps: int = 40
    dynamics_horizon: int = 3
    dynamics_lr: float = 1e-4
    seed: int = 110606
    grpo_train: list[int] = field(default_factory=lambda: list(range(11066000, 11066004)))
    wm_fit: list[int] = field(default_factory=lambda: list(range(11067000, 11067004)))
    wm_validate: list[int] = field(default_factory=lambda: list(range(11068000, 11068004)))
    confirm: list[int] = field(default_factory=lambda: list(range(11069000, 11069004)))
    test: list[int] = field(default_factory=lambda: list(range(11070000, 11070010)))

    def splits(self):
        return {key: getattr(self, key) for key in ('grpo_train', 'wm_fit', 'wm_validate', 'confirm', 'test')}

    def check(self):
        seeds = sum(self.splits().values(), [])
        if (any(not x for x in self.splits().values()) or len(set(seeds)) != len(seeds)
                or set(seeds) & set(range(10290200, 10290300))
                or any(type(x) is not int or x < 0 for x in seeds)):
            raise ValueError('Nonempty, disjoint fresh seed splits required')
        if not (1 <= self.horizon_packets <= 112 and 2 <= self.group_size <= 8
                and 1 <= self.grpo_iterations <= 10 and 1 <= self.grpo_epochs <= 4
                and 1 <= self.dynamics_steps <= 200 and 1 <= self.dynamics_horizon <= 10
                and 0 < self.policy_lr <= .01 and 0 < self.dynamics_lr <= .01
                and 0 <= self.kl_beta <= 1 and 0 < self.residual_radius <= .05
                and 0 < self.residual_std <= 2 and all(len(s) <= 100 for s in self.splits().values())):
            raise ValueError('Outside bounded pilot training budget')

    def budget(self):
        return dict(simulator_episodes=2 * (len(self.wm_fit) + len(self.wm_validate))
                    + 5 * (len(self.confirm) + len(self.test)),
                    grpo_imagined_episodes=3 * self.grpo_iterations * len(self.grpo_train) * self.group_size,
                    final_prediction_episodes=2 * 5 * len(self.test),
                    seconds_per_episode=.2 * self.horizon_packets)


def run_training(simulator, original_env, baseline, original_module, output, config=None,
                 provenance=None, *, require_real=True):
    """All new success rates come from simulator episodes; no test-time selection."""
    config = config or TrainingConfig()
    config.check()
    if require_real:
        from .mujoco_backend import MuJoCoTube
        if not isinstance(simulator, MuJoCoTube) or simulator.evidence_kind != 'mujoco':
            raise TypeError('Real training requires the original AutoBio MuJoCo adapter')
    evidence = simulator.evidence_kind
    validate_module(original_module)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    output.mkdir(parents=True)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    write_json(output / 'protocol.json', dict(config=asdict(config), budget=config.budget(),
                                            evidence_kind=evidence, provenance=provenance or {}))

    def initials(split):
        result = []
        for seed in config.splits()[split]:
            initial = simulator.initial(seed)
            initial.observation.validate('simulator')
            if initial.observation.official_success or initial.observation.terminated:
                raise ValueError('Already successful/terminal initial state; protocol invalid')
            directory = output / 'initials' / split
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f'{seed}.npz', **initial.snapshot,
                                policy_state=initial.observation.policy_state)
            write_json(directory / f'{seed}.json', dict(seed=seed, snapshot_sha256=initial.digest,
                       height=initial.observation.height, contact=initial.observation.contact))
            result.append(initial)
        return result

    train_starts = initials('grpo_train')
    bridge = check_bridge(simulator, original_env, baseline, config.grpo_train[0])
    baseline.reset()
    packet = baseline.packet(copy.deepcopy(train_starts[0].observation))
    baseline.reset()
    repeat_packet = baseline.packet(copy.deepcopy(train_starts[0].observation))
    if not np.array_equal(packet, repeat_packet):
        raise ValueError('Frozen baseline is not deterministic after reset; check eval/chunk state')
    bridge['baseline_reset_repeatable'] = True
    packets = [packet]
    for sign in (-1, 1):
        probe = np.asarray(packet).copy()
        probe[:, :6] += sign * .005 * (simulator.bounds[:6, 1] - simulator.bounds[:6, 0])
        packets.append(np.clip(probe, simulator.bounds[:, 0], simulator.bounds[:, 1]))
    bridge['trainable_equivalence'] = check_equivalence(original_env, original_module, train_starts[0], packets)
    write_json(output / 'bridge_check.json', bridge)
    model0 = TorchLatentEnv(copy.deepcopy(original_module))
    pi_init = ResidualPolicy(original_module.state_mean.tolist(), original_module.state_scale.tolist(),
                            simulator.bounds, config.residual_radius, config.residual_std)
    trainer = GRPOTrainer(copy.deepcopy(pi_init), lr=config.policy_lr, beta=config.kl_beta,
                          epochs=config.grpo_epochs, seed=config.seed)
    trainer.save(output / 'pi_init.pt')
    stage_a = train_imagined(trainer, model0, baseline, train_starts, simulator.bounds,
                              config.horizon_packets, config.group_size, config.grpo_iterations, output / 'grpo_a')
    pi1 = copy.deepcopy(trainer.policy)

    def run_episode(env, policy, initial, split, name, source):
        # Common random numbers across policies AND model/simulator paired rollouts.
        ep, _ = policy_episode(env, baseline, policy, initial, simulator.bounds, config.horizon_packets,
                               source, split, evidence if source == 'simulator' else 'imagined',
                               config.seed + initial.seed, name)
        ep.save(output / 'episodes' / split / name / source / f'{initial.seed}.npz')
        return ep

    data = {}
    for split in ('wm_fit', 'wm_validate'):
        data[split] = []
        for initial in initials(split):
            for name, policy in [('pi0', None), ('pi1', pi1)]:
                data[split].append(run_episode(simulator, policy, initial, split, name, 'simulator'))
    model1_module, model_report = fit_dynamics(original_module, data['wm_fit'], data['wm_validate'],
                    output / 'latent_fit', steps=config.dynamics_steps, horizon=config.dynamics_horizon,
                    lr=config.dynamics_lr, seed=config.seed + 1, evidence_kind=evidence)
    model1 = TorchLatentEnv(model1_module)
    # Both arms start from exactly the same pi1/optimizer/reference/RNG state.
    control = GRPOTrainer.load(output / 'grpo_a' / 'checkpoint.pt')
    stage_b = train_imagined(trainer, model1, baseline, train_starts, simulator.bounds,
                              config.horizon_packets, config.group_size, config.grpo_iterations, output / 'grpo_b')
    frozen_control = train_imagined(control, model0, baseline, train_starts, simulator.bounds,
                              config.horizon_packets, config.group_size, config.grpo_iterations, output / 'frozen_t0_control')
    policies = dict(pi0=None, pi_init=pi_init, pi1=pi1,
                    pi2=copy.deepcopy(trainer.policy), frozen_t0=copy.deepcopy(control.policy))
    confirmation = {name: [] for name in policies}
    for initial in initials('confirm'):
        for name, policy in policies.items():
            confirmation[name].append(run_episode(simulator, policy, initial, 'confirm', name, 'simulator').success)
    # Independent physical confirmation decides promotion. Ties retain pi0.
    selected = 'pi0'
    for name in ('pi1', 'pi2', 'frozen_t0'):
        if sum(confirmation[name]) > sum(confirmation[selected]):
            selected = name
    selection = dict(selected_policy=selected, basis='confirm_simulator_success_count',
                     confirmation=confirmation, model_update_accepted=model_report['accepted'])
    write_json(output / 'frozen_selection.json', selection)
    for name, policy in policies.items():
        if policy is not None:
            torch.save(dict(spec=policy.spec(), state=policy.state_dict()), output / f'{name}_frozen.pt')
    # First access to final test initial states occurs after all weights/selection freeze.
    actual = {name: [] for name in policies}
    predictions = {name: {'T0': [], 'T1_selected': []} for name in policies}
    for initial in initials('test'):
        for name, policy in policies.items():
            actual[name].append(run_episode(simulator, policy, initial, 'test', name, 'simulator').success)
            for version, env in [('T0', model0), ('T1_selected', model1)]:
                predictions[name][version].append(run_episode(env, policy, initial, 'test', name + '_' + version, 'world_model').success)
    pairs = [dict(seed=seed, policy=name, actual=bool(actual[name][i]),
                  T0=int(predictions[name]['T0'][i]), T1_selected=int(predictions[name]['T1_selected'][i]))
             for i, seed in enumerate(config.test) for name in policies]
    write_json(output / 'final_pairs.json', pairs)
    optimizer_steps = sum(len(log['losses']) for log in stage_a + stage_b + frozen_control)
    result = dict(status='COMPLETE', evidence_kind=evidence, config=asdict(config),
                  selection=selection, actual_success_labels=actual,
                  policy_comparisons={name: paired_difference(actual['pi0'], labels) for name, labels in actual.items() if name != 'pi0'},
                  learning_vs_initial=paired_difference(actual['pi_init'], actual['pi1']),
                  model_update_vs_extra_training=paired_difference(actual['frozen_t0'], actual['pi2']),
                  selected_policy_comparison=paired_difference(actual['pi0'], actual[selected]),
                  heldout_model_errors={name: {version: confusion(actual[name], labels)
                      for version, labels in versions.items()} for name, versions in predictions.items()},
                  latent_update=model_report, grpo_a=stage_a, grpo_b=stage_b, frozen_control=frozen_control,
                  grpo_optimizer_steps=optimizer_steps, grpo_performed=optimizer_steps > 0,
                  model_dynamics_fitted=bool(model_report['changed_dynamics_parameters']),
                  model_dynamics_update_selected=model_report['accepted'],
                  final_policy_ranking={version: ranking_metrics(pairs, version) for version in ('T0', 'T1_selected')},
                  real_success_improved=sum(actual[selected]) > sum(actual['pi0']))
    write_json(output / 'result.json', result)
    return result
