"""Actual optimizer tests on labelled synthetic fixtures, NOT AutoBio results."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from torch import nn
from dyna05.core import rollout, Candidate
from dyna05.grpo import (ResidualPolicy, GRPOTrainer, group_advantages, clipped_objective,
                         pack_decisions, policy_episode)
from dyna05.latent import (TorchLatentEnv, fit_dynamics, check_equivalence, window_loss,
                           evaluate_dynamics)
from dyna05.training import TrainingConfig, run_training
from test_dyna05 import FixtureSimulator, FixtureController


class ToyLatent(nn.Module):
    """Known biased one-state fixture, with a genuinely trainable transition."""
    dynamics_parameter_names = ('drift', 'gain')
    readout_parameter_names = ()

    def __init__(self, drift=.01):
        super().__init__()
        self.drift = nn.Parameter(torch.tensor(drift, dtype=torch.float64))
        self.gain = nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        self.register_buffer('state_mean', torch.tensor([.9], dtype=torch.float64))
        self.register_buffer('state_scale', torch.tensor([.05], dtype=torch.float64))
        self.register_buffer('contact', torch.tensor(12., dtype=torch.float64))

    def encode(self, x):
        return x

    def transition(self, z, actions):
        return z + self.drift + self.gain * actions[:, :, 0].mean(-1, keepdim=True)

    def decode(self, z):
        return dict(policy_state=z, height=z[:, 0], contact_logits=self.contact.expand(len(z)))


def policy():
    torch.manual_seed(4)
    return ResidualPolicy([.9], [.05], FixtureSimulator.bounds)


def sampled_decisions(p, rng, n=16):
    episodes, rewards = [], []
    for _ in range(n):
        packet, decision = p.sample([.91], np.zeros((10, 7)), rng)
        episodes.append([decision])
        rewards.append(float(packet[0, 0] > 0))
    return episodes, torch.tensor([rewards], dtype=torch.float64)


def test_normalized_advantages_and_constant_groups():
    advantage, zeros = group_advantages(torch.tensor([[0., 1., 1., 0.], [1., 1., 1., 1.]]))
    torch.testing.assert_close(advantage[0], torch.tensor([-1., 1., 1., -1.]))
    assert advantage[1].count_nonzero() == 0 and zeros.tolist() == [False, True]


def test_surrogate_positive_negative_clipping_and_padding():
    logp = torch.tensor([[np.log(1.5), 99.], [np.log(.5), 99.]], dtype=torch.float64)
    # Padding is finite, but must never affect either trajectory's denominator.
    result = clipped_objective(logp, torch.zeros_like(logp), torch.tensor([1., -1.]),
                               torch.tensor([[1., 0.], [1., 0.]]))
    assert float(result) == pytest.approx(.2)


def test_exact_executed_density_and_gripper_preservation():
    p = policy()
    base = np.zeros((10, 7)); base[:, 6] = .8
    packet, trace = p.sample([.91], base, torch.Generator().manual_seed(6))
    center, half = p.interval(torch.tensor(base))
    torch.testing.assert_close(torch.tensor(packet[:, :6]), (center + half * trace['u'].tanh()).expand(10, 6))
    expected = torch.distributions.Normal(p(trace['state']), p.std).log_prob(trace['u'])
    expected -= half.log() + (1 - trace['u'].tanh().square()).log()
    torch.testing.assert_close(trace['old_logp'], expected.sum())
    np.testing.assert_array_equal(packet[:, 6], base[:, 6])


def test_feasible_intersection_and_deterministic_axes():
    p = policy()
    rng = torch.Generator().manual_seed(20)
    for slack in (0., 1e-13):
        base = np.zeros((10, 7)); base[0, 0] = 1.; base[1, 0] = -1. + slack; base[:, 1] = 1.
        center, half = p.interval(torch.tensor(base))
        assert half[0] == 0
        for _ in range(100):
            packet, trace = p.sample([.91], base, rng)
            assert trace['half'][0] == 0 and torch.isfinite(trace['old_logp'])
            np.testing.assert_array_equal(packet[:, 0], base[:, 0] + float(center[0]))
            assert (packet <= 1).all() and (packet >= -1).all()


def test_out_of_bound_baseline_rejected_without_clipping():
    with pytest.raises(ValueError, match='Frozen controller'):
        policy().sample([.91], np.full((10, 7), 1.01), torch.Generator())


def test_reference_kl_matches_gaussian_and_reference_frozen():
    p = policy(); trainer = GRPOTrainer(p)
    with torch.no_grad():
        p.net[-1].bias[0] = .3
    kl = p.reference_kl(torch.tensor([[.91]], dtype=torch.float64), torch.ones(1, 6), trainer.reference)
    assert float(kl.detach()) == pytest.approx(.3**2 / (2 * .5**2))
    assert all(not x.requires_grad for x in trainer.reference.parameters())


def test_old_log_prob_frozen_and_zero_variance_skipped():
    p = policy(); trainer = GRPOTrainer(p)
    decisions, rewards = sampled_decisions(p, trainer.generator)
    old = pack_decisions(decisions)['old_logp'].clone()
    before = copy.deepcopy(p.state_dict())
    result = trainer.update(decisions, torch.ones_like(rewards))
    assert result['optimizer_steps'] == 0 and result['parameter_delta_l2'] == 0
    for name, value in before.items():
        torch.testing.assert_close(p.state_dict()[name], value, rtol=0, atol=0)
    trainer.update(decisions, rewards)
    torch.testing.assert_close(pack_decisions(decisions)['old_logp'], old, rtol=0, atol=0)


def test_tiny_grpo_optimization_increases_known_action_probability():
    p = policy(); trainer = GRPOTrainer(p, lr=.01, beta=0, epochs=2)
    reference = copy.deepcopy(trainer.reference.state_dict())
    for _ in range(12):
        decisions, rewards = sampled_decisions(p, trainer.generator, 64)
        log = trainer.update(decisions, rewards)
        assert np.isfinite(log['grad_norm'])
    # The synthetic objective is P(a_0 > 0); its exact Gaussian probability is known.
    mean = p(torch.tensor([.91], dtype=torch.float64))[0].detach()
    probability = torch.distributions.Normal(0., 1.).cdf(mean / p.std)
    assert probability > .85 and trainer.optimizer_steps > 0
    for name, value in reference.items():
        torch.testing.assert_close(trainer.reference.state_dict()[name], value, rtol=0, atol=0)


def test_checkpoint_resumes_optimizer_and_sampling_exactly(tmp_path):
    trainer = GRPOTrainer(policy())
    trainer.update(*sampled_decisions(trainer.policy, trainer.generator))
    trainer.save(tmp_path / 'checkpoint.pt')
    resumed = GRPOTrainer.load(tmp_path / 'checkpoint.pt')
    left = sampled_decisions(trainer.policy, trainer.generator)
    right = sampled_decisions(resumed.policy, resumed.generator)
    torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
    assert trainer.update(*left) == resumed.update(*right)
    for a, b in zip(trainer.policy.parameters(), resumed.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def episodes(split, seeds):
    sim = FixtureSimulator()
    return [rollout(sim, FixtureController(), sim.initial(seed), Candidate('pi0', (0.,)*7),
                    sim.bounds, 4, 'simulator', split, sim.evidence_kind) for seed in seeds]


def test_latent_optimizer_updates_dynamics_and_improves_heldout(tmp_path):
    original = ToyLatent(.02)
    saved = copy.deepcopy(original.state_dict())
    updated, report = fit_dynamics(original, episodes('wm_fit', [1, 2, 3]),
                        episodes('wm_validate', [4, 5]), tmp_path, steps=100, lr=.002,
                        evidence_kind='synthetic_test')
    assert report['accepted'] and report['changed_dynamics_parameters'] == ['drift']
    assert report['after']['open_loop_height_mse'] < report['before']['open_loop_height_mse']
    assert not torch.equal(updated.drift, original.drift)
    for name, value in saved.items():
        torch.testing.assert_close(original.state_dict()[name], value, rtol=0, atol=0)


def test_lower_prediction_loss_does_not_override_worse_success_judgments(tmp_path):
    _, report = fit_dynamics(ToyLatent(.02), episodes('wm_fit', [1, 2, 3]),
                        episodes('wm_validate', [4, 5]), tmp_path, steps=50, lr=.002,
                        evidence_kind='synthetic_test')
    assert report['after']['one_step_loss'] < report['before']['one_step_loss']
    assert report['after']['reward_errors']['fn'] > report['before']['reward_errors']['fn']
    assert not report['accepted']


@pytest.mark.parametrize('change', ['imagined', 'test', 'overlap', 'evidence'])
def test_latent_rejects_leakage_and_wrong_evidence(tmp_path, change):
    train, val = episodes('wm_fit', [1]), episodes('wm_validate', [2])
    if change == 'imagined': train[0].source = 'world_model'
    if change == 'test': train[0].split = 'test'
    if change == 'overlap': val[0].seed = 1
    if change == 'evidence': train[0].evidence_kind = 'mujoco'
    with pytest.raises(ValueError):
        fit_dynamics(ToyLatent(), train, val, tmp_path, evidence_kind='synthetic_test')


def test_recursive_loss_does_not_teacher_force_future_states():
    model = ToyLatent()
    ep = episodes('wm_fit', [1])[0]
    seen = []
    original = model.transition
    def spy(z, actions):
        seen.append(z.detach().clone())
        return original(z, actions)
    model.transition = spy
    window_loss(model, ep, 0, 3).backward()
    assert len(seen) == 3 and model.drift.grad != 0
    assert float(seen[1][0, 0]) == pytest.approx(ep.states[0, 0] + .01)
    assert float(seen[1][0, 0]) != pytest.approx(ep.states[1, 0])


def test_wrapper_must_match_original_recurrent_predictions():
    sim = FixtureSimulator(); initial = sim.initial(1)
    good, bad = ToyLatent(), ToyLatent(.03)
    with pytest.raises(ValueError, match='does not reproduce'):
        check_equivalence(TorchLatentEnv(good), bad, initial, [np.zeros((10, 7))] * 3)


@pytest.mark.parametrize('change', [dict(test=[11066000]), dict(test=[10290201]), dict(group_size=1),
                                   dict(grpo_iterations=0), dict(horizon_packets=113)])
def test_training_config_guards(change):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **change).check()


def test_real_runner_rejects_synthetic_environment(tmp_path):
    module = ToyLatent()
    with pytest.raises(TypeError, match='original AutoBio'):
        run_training(FixtureSimulator(), TorchLatentEnv(module), FixtureController(), module, tmp_path / 'no')


def test_entire_training_cycle_freezes_before_test_and_records_real_source(tmp_path):
    config = TrainingConfig(horizon_packets=4, grpo_iterations=1, dynamics_steps=5,
                            grpo_train=[1, 2], wm_fit=[3, 4], wm_validate=[5, 6], confirm=[7, 8], test=[9, 10])
    output = tmp_path / 'cycle'
    class WatchedSimulator(FixtureSimulator):
        def initial(self, seed):
            if seed in config.test:
                assert (output / 'frozen_selection.json').is_file()
                assert (output / 'pi2_frozen.pt').is_file()
            return super().initial(seed)
    sim, model = WatchedSimulator(), ToyLatent()
    result = run_training(sim, TorchLatentEnv(model), FixtureController(), model, output, config, require_real=False)
    assert result['evidence_kind'] == 'synthetic_test' and result['status'] == 'COMPLETE'
    assert set(result['actual_success_labels']) == {'pi0', 'pi_init', 'pi1', 'pi2', 'frozen_t0'}
    assert all(x['reward_source'] == 'world_model' for x in result['grpo_a'])
    assert result['policy_comparisons']['pi2']['n'] == 2
    assert (output / 'latent_fit/candidate.pt').is_file()
    # When the fit is rejected, equal-compute continuation on identical T0 must match exactly.
    if not result['latent_update']['accepted']:
        a = torch.load(output / 'pi2_frozen.pt', weights_only=True)
        b = torch.load(output / 'frozen_t0_frozen.pt', weights_only=True)
        for key in a['state']:
            torch.testing.assert_close(a['state'][key], b['state'][key], rtol=0, atol=0)
    stored = json.loads((output / 'result.json').read_text())
    assert stored['actual_success_labels'] == result['actual_success_labels']
