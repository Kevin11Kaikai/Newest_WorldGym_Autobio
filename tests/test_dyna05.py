"""Synthetic contract tests. None of these outcomes are AutoBio experiments."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from dyna05.core import (Observation, InitialState, Candidate, Config, RewardCorrection,
                         generate_candidates, rollout, run_cycle, confusion,
                         paired_difference, ranking_metrics, features)
from dyna05.core import check_bridge
from dyna05.assets import preflight
from dyna05.audit import audit_stage4


class FixtureController:
    def reset(self):
        self.calls = 0

    def packet(self, observation):
        self.calls += 1
        result = np.zeros((10, 7))
        # Read the evolving state: this is a feedback policy, not action replay.
        result[:, 1] = (observation.height - .9) * .01
        return result


class FixtureSimulator:
    evidence_kind = 'synthetic_test'
    bounds = np.array([[-1., 1.]] * 7)

    def __init__(self):
        self.started = []

    def initial(self, seed):
        self.started.append(seed)
        h = .89 + .03 * np.random.default_rng(seed).uniform()
        return InitialState(seed, {'height': np.array([h])}, Observation(np.array([h]), h, 1., False))

    def reset(self, initial):
        self.h = float(initial.snapshot['height'][0])
        return Observation(np.array([self.h]), self.h, 1., self.h > .925)

    def step(self, packet):
        self.h += .002 + .7 * packet[:, 0].mean()
        return Observation(np.array([self.h]), self.h, 1., self.h > .925)


class OptimisticFixtureModel:
    def reset(self, observation):
        assert observation.official_success is None
        self.h = observation.height
        self.seen = [self.h]
        return Observation(np.array([self.h]), self.h, 1.)

    def step(self, packet):
        self.h += .025 + .3 * packet[:, 0].mean()
        self.seen.append(self.h)
        return Observation(np.array([self.h]), self.h, 1.)


def small_config():
    return Config(horizon_packets=4, candidates=3, verify_top_k=2,
                  collect=list(range(10, 30)), validate=list(range(30, 50)),
                  search=list(range(50, 54)), confirm=list(range(54, 62)),
                  test=list(range(70, 90)))


def fixed_candidates():
    return [Candidate('baseline', (0.,) * 7),
            Candidate('candidate_good', (.01, 0., 0., 0., 0., 0., 0.)),
            Candidate('candidate_bad', (-.01, 0., 0., 0., 0., 0., 0.))]


def test_historical_counts():
    audit = audit_stage4(Path(__file__).resolve().parents[1])
    assert audit['new_simulator_runs'] == 0
    assert [audit['metrics'][k] for k in ('tp', 'fn', 'fp', 'tn')] == [86, 4, 9, 1]
    assert audit['metrics']['failure_recall'] == .1


def test_candidate_reproducibility_and_action_bounds():
    a = generate_candidates()
    assert a == generate_candidates()
    assert len({c.residual_fraction for c in a}) == len(a)
    assert all(c.residual_fraction[6] == 0 for c in a)
    executed, clipped = a[1].apply(np.full((10, 7), 1.), FixtureSimulator.bounds)
    assert np.max(executed) <= 1 and np.min(executed) >= -1
    assert clipped > 0


@pytest.mark.parametrize('bad', [np.zeros((50, 7)), np.full((10, 7), np.nan)])
def test_action_schema_rejected(bad):
    with pytest.raises(ValueError):
        fixed_candidates()[0].apply(bad, FixtureSimulator.bounds)


@pytest.mark.parametrize('source,obs', [
    ('simulator', Observation(np.array([0.]), .9, 1., None)),
    ('world_model', Observation(np.array([0.]), .9, 1., True)),
    ('world_model', Observation(np.array([0.]), .9, 2., None)),
])
def test_labels_and_contact_cannot_cross_sources(source, obs):
    with pytest.raises(ValueError):
        obs.validate(source)


def test_model_success_requires_simultaneous_height_and_contact():
    class CrossingModel(OptimisticFixtureModel):
        def step(self, packet):
            self.h += .1
            return Observation(np.array([self.h]), self.h, 0.)
    sim = FixtureSimulator()
    episode = rollout(CrossingModel(), FixtureController(), sim.initial(1), fixed_candidates()[0],
                      sim.bounds, 4, 'world_model', 'search', 'synthetic_test')
    assert episode.height.max() > .925 and episode.contact.max() == 1
    assert not episode.success


def test_reset_preserves_fixed_state_and_closed_loop_feedback():
    sim, controller = FixtureSimulator(), FixtureController()
    initial = sim.initial(1)
    a = rollout(sim, controller, initial, fixed_candidates()[1], sim.bounds, 4,
                'simulator', 'confirm', sim.evidence_kind)
    b = rollout(sim, controller, initial, fixed_candidates()[1], sim.bounds, 4,
                'simulator', 'confirm', sim.evidence_kind)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.states, b.states)
    assert a.initial_hash == b.initial_hash == initial.digest
    assert a.actions[0, 0, 1] != a.actions[-1, 0, 1]


def test_terminal_transitions_and_saved_alignment(tmp_path):
    class EarlyStop(FixtureSimulator):
        def step(self, packet):
            obs = super().step(packet)
            obs.terminated = True
            return obs
    sim = EarlyStop()
    ep = rollout(sim, FixtureController(), sim.initial(1), fixed_candidates()[0], sim.bounds,
                 4, 'simulator', 'collect', sim.evidence_kind)
    assert ep.actions.shape == (1, 10, 7) and ep.states.shape[0] == 2
    ep.save(tmp_path / 'episode.npz')
    with np.load(tmp_path / 'episode.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['next_states'], ep.states[1:])
        assert not saved['truncated']
        assert json.loads(str(saved['metadata']))['evidence_kind'] == 'synthetic_test'


@pytest.mark.parametrize('replacement', [dict(test=[10]), dict(test=[10290201]), dict(test=[90, 90]),
                                      dict(horizon_packets=113)])
def test_split_and_budget_guards(replacement):
    with pytest.raises(ValueError):
        replace(small_config(), **replacement).check()


@pytest.mark.parametrize('split,label', [('test', 'simulator'), ('validate', 'simulator'), ('collect', 'world_model')])
def test_correction_rejects_test_or_imagined_labels(split, label):
    with pytest.raises(ValueError):
        RewardCorrection.fit([dict(split=split, label_source=label, features=[0.], actual=1)])


def test_correction_requires_diverse_real_feedback():
    with pytest.raises(ValueError):
        RewardCorrection.fit([dict(split='collect', label_source='simulator', features=[0.], actual=1)])


def test_reward_head_reload(tmp_path):
    rows = [dict(split='collect', label_source='simulator', features=[x], actual=x > 0) for x in [-2, -1, 1, 2]]
    head = RewardCorrection.fit(rows)
    path = tmp_path / 'head.json'
    head.save(path)
    restored = RewardCorrection.load(path)
    np.testing.assert_array_equal(head.predict([[-1], [1]]), restored.predict([[-1], [1]]))
    assert restored.predict([[1]])[0] > restored.predict([[-1]])[0]


def test_ranking_ties_are_not_positive_evidence():
    rows = [dict(seed=1, predicted=1, actual=x) for x in [0, 1]]
    assert ranking_metrics(rows, 'predicted')['pairwise_accuracy'] == .5
    assert ranking_metrics(rows, 'predicted')['zero_variance_groups'] == 1


def test_paired_metrics_not_unpaired_counts():
    r = paired_difference([0, 0, 1, 1], [1, 1, 0, 1])
    assert r['wins'] == 2 and r['losses'] == 1 and r['difference'] == .25
    assert paired_difference([0, 1], [0, 1])['exact_mcnemar_p'] == 1
    assert confusion([1, 1], [1, 1])['failure_recall'] is None


def test_complete_cycle_and_no_final_test_leakage(tmp_path):
    cfg = small_config()
    sim = FixtureSimulator()
    # Final start creation is observable and must follow frozen policy selection.
    original_initial = sim.initial
    def initial(seed):
        if seed in cfg.test:
            assert (tmp_path / 'run/selected_policy.json').exists()
        return original_initial(seed)
    sim.initial = initial
    report = run_cycle(sim, OptimisticFixtureModel(), FixtureController(), tmp_path / 'run',
                       cfg, fixed_candidates())
    assert report['evidence_kind'] == 'synthetic_test'
    assert report['selected_policy'] == 'candidate_good'
    assert report['paired_policy_evaluation']['after_successes'] == len(cfg.test)
    assert not report['model_dynamics_updated'] and not report['grpo_performed']
    pairs = json.loads((tmp_path / 'run/pairs.json').read_text())
    assert all(r['label_source'] == 'simulator' for r in pairs)
    correction = report['model_correction']
    if correction['accepted']:
        assert correction['after']['fn'] <= correction['before']['fn']
        assert correction['after']['fp'] <= correction['before']['fp']
        assert correction['after_brier'] < correction['before_brier']
    selected_rows = [r for r in pairs if r['split'] == 'test' and r['candidate'] == 'candidate_good']
    assert len(selected_rows) == len(cfg.test)
    assert set(cfg.search).isdisjoint({r['seed'] for r in pairs})
    with pytest.raises(FileExistsError):
        run_cycle(sim, OptimisticFixtureModel(), FixtureController(), tmp_path / 'run', cfg, fixed_candidates())


def test_false_model_success_never_promotes_a_failed_policy(tmp_path):
    class AlwaysFails(FixtureSimulator):
        def step(self, packet):
            self.h += 0
            return Observation(np.array([self.h]), self.h, 1., False)
    result = run_cycle(AlwaysFails(), OptimisticFixtureModel(), FixtureController(), tmp_path / 'veto',
                       small_config(), fixed_candidates())
    assert result['selected_policy'] == 'baseline'
    assert result['paired_policy_evaluation']['after_successes'] == 0
    assert not result['model_correction']['accepted']


def test_validated_correction_can_be_used_in_search(tmp_path):
    class SeparableFeedback(FixtureSimulator):
        def step(self, packet):
            self.h += .7 * packet[:, 0].mean()
            return Observation(np.array([self.h]), self.h, 1., self.h > .925)
    result = run_cycle(SeparableFeedback(), OptimisticFixtureModel(), FixtureController(), tmp_path / 'accepted',
                       small_config(), fixed_candidates())
    assert result['model_correction']['accepted']
    assert (tmp_path / 'accepted/reward_correction_selected.json').exists()
    assert result['model_correction']['after']['fp'] < result['model_correction']['before']['fp']
    assert result['selected_policy'] == 'candidate_good'


def test_preflight_blocks_missing_assets():
    path = Path(__file__).resolve().parents[1] / 'configs/notebook05_assets.example.json'
    report = preflight(path)
    assert report['status'] == 'BLOCKED'
    assert any('policy_checkpoint' in b for b in report['blockers'])


def test_mujoco_packet_adapter_contract_without_physics(monkeypatch):
    # Test scheduling and state restore with an explicitly fake MuJoCo API.
    # This does not validate real physics or native plugin compatibility.
    import sys
    from types import SimpleNamespace
    from dyna05.mujoco_backend import MuJoCoTube
    class Task:
        def __init__(self):
            self.model = SimpleNamespace(actuator_ctrlrange=FixtureSimulator.bounds,
                                         opt=SimpleNamespace(timestep=.002))
            self.data = SimpleNamespace(time=0., ctrl=np.zeros(7), ncon=0, contact=[],
                                        warning=SimpleNamespace(number=np.zeros(1)))
            self.object = SimpleNamespace(get_body_pose=lambda data: SimpleNamespace(pos=[0., 0., .9]))
            self.steps = 0
        def reset(self, seed):
            self.data.time = 0.
            self.data.ctrl[:] = 0.
        def check(self):
            return False
        def step_and_log(self, info):
            self.steps += 1
            self.data.time += .002
        def finish(self):
            pass
    def get_state(model, data, out, spec):
        out[:] = np.r_[data.time, data.ctrl]
    def set_state(model, data, arr, spec):
        data.time = float(arr[0])
        data.ctrl[:] = arr[1:]
    fake = SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
                           mjtObj=SimpleNamespace(mjOBJ_GEOM=1),
                           mj_name2id=lambda model, kind, name: 0 if 'leftfinger' in name else 1,
                           mj_stateSize=lambda model, spec: 8,
                           mj_getState=get_state, mj_setState=set_state,
                           mj_forward=lambda model, data: None)
    monkeypatch.setitem(sys.modules, 'mujoco', fake)
    task = Task()
    adapter = MuJoCoTube(task, lambda task: task.data.ctrl)
    initial = adapter.initial(5)
    adapter.step(np.full((10, 7), .1))
    assert task.steps == 100
    assert np.isclose(task.data.time, .2)
    adapter.reset(initial)
    assert task.data.time == 0
    np.testing.assert_array_equal(task.data.ctrl, np.zeros(7))
    with pytest.raises(ValueError, match='Out-of-bound'):
        adapter.step(np.full((10, 7), 2.))
    task.data.warning.number[0] = 1
    with pytest.raises(RuntimeError, match='warning'):
        adapter.step(np.zeros((10, 7)))


def test_initial_state_alias_rejected(tmp_path):
    class Aliased(FixtureSimulator):
        def initial(self, seed):
            result = super().initial(1)
            result.seed = seed
            return result
    with pytest.raises(ValueError, match='Repeated physical initial state'):
        run_cycle(Aliased(), OptimisticFixtureModel(), FixtureController(), tmp_path / 'alias',
                   small_config(), fixed_candidates())


def test_bridge_action_conditioning_gate():
    result = check_bridge(FixtureSimulator(), OptimisticFixtureModel(), FixtureController(), 1)
    assert result['reset_repeatable'] and result['max_action_response'] > 0
    class ActionIndependent(OptimisticFixtureModel):
        def step(self, packet):
            return super().step(np.zeros_like(packet))
    with pytest.raises(ValueError, match='action dependence'):
        check_bridge(FixtureSimulator(), ActionIndependent(), FixtureController(), 1)
