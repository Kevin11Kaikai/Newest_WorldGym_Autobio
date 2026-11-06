"""One Dyna-style iteration. The core never manufactures physical outcomes.

The private Stage 4 architecture is deliberately behind a small bridge contract.
The executable loop, candidate generation, update, and evaluation live here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
import copy
import hashlib
import json
import math
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise ValueError('Object arrays are not portable snapshots')
        digest.update(key.encode())
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass
class Observation:
    # Exact original controller input, supplied by the Stage 4 bridge.
    policy_state: np.ndarray
    height: float
    contact: float
    official_success: bool | None = None
    terminated: bool = False

    def validate(self, source):
        x = np.asarray(self.policy_state)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('policy_state must be a finite 1-D vector')
        if not np.isfinite([self.height, self.contact]).all() or not 0 <= self.contact <= 1:
            raise ValueError('Invalid height/contact')
        if source == 'simulator' and not isinstance(self.official_success, (bool, np.bool_)):
            raise ValueError('Simulator must provide its own official success label')
        if source == 'world_model' and self.official_success is not None:
            raise ValueError('A model prediction cannot carry an official simulator label')


@dataclass
class InitialState:
    seed: int
    snapshot: dict[str, np.ndarray]
    observation: Observation

    @property
    def digest(self):
        return fingerprint(self.snapshot)


class Simulator(Protocol):
    evidence_kind: str  # "mujoco" for real execution; fixtures use "synthetic_test".
    bounds: np.ndarray  # (7, 2), physical actuator command units.
    def initial(self, seed: int) -> InitialState: ...
    def reset(self, initial: InitialState) -> Observation: ...
    def step(self, packet: np.ndarray) -> Observation: ...


class WorldModel(Protocol):
    def reset(self, initial_observation: Observation) -> Observation: ...
    def step(self, packet: np.ndarray) -> Observation: ...


class Controller(Protocol):
    def reset(self) -> None: ...
    def packet(self, observation: Observation) -> np.ndarray: ...


def check_bridge(simulator, model, controller, seed):
    """Development-only interface smoke test; executes no physical action."""
    initial = simulator.initial(seed)
    initial.observation.validate('simulator')
    controller.reset()
    packet = np.asarray(controller.packet(copy.deepcopy(initial.observation)))
    baseline = Candidate('baseline', (0.,) * 7)
    packet, _ = baseline.apply(packet, np.asarray(simulator.bounds))

    def predict(command):
        obs = copy.deepcopy(initial.observation)
        obs.official_success = None
        model.reset(obs).validate('world_model')
        prediction = model.step(command.copy())
        prediction.validate('world_model')
        if np.asarray(prediction.policy_state).shape != np.asarray(initial.observation.policy_state).shape:
            raise ValueError('Model and simulator controller-state dimensions differ')
        return np.r_[prediction.policy_state, prediction.height, prediction.contact]

    before = predict(packet)
    repeat = predict(packet)
    if not np.allclose(before, repeat, atol=1e-7, rtol=1e-6):
        raise ValueError('Model reset/step is not deterministic under the current protocol')
    # Probe both directions, since the baseline may lie at an actuator bound.
    changes = []
    for direction in (-1., 1.):
        perturbed = packet.copy()
        width = simulator.bounds[:, 1] - simulator.bounds[:, 0]
        perturbed[:, :6] += direction * .01 * width[:6]
        perturbed = np.clip(perturbed, simulator.bounds[:, 0], simulator.bounds[:, 1])
        changes.append(float(np.max(np.abs(predict(perturbed) - before))))
    if max(changes) <= 1e-10:
        raise ValueError('No detectable action dependence; repair the original model interface')
    return {'development_seed': seed, 'reset_repeatable': True,
            'max_action_response': max(changes), 'physical_action_steps': 0}


@dataclass(frozen=True)
class Candidate:
    name: str
    # One fixed bounded residual per actuator; the underlying policy is closed-loop.
    residual_fraction: tuple[float, ...]

    def apply(self, baseline_packet, bounds):
        packet = np.asarray(baseline_packet, dtype=np.float64)
        if packet.shape != (10, 7) or not np.isfinite(packet).all():
            raise ValueError('Controller must emit one physical 10 x 7 packet')
        width = bounds[:, 1] - bounds[:, 0]
        raw = packet + np.asarray(self.residual_fraction) * width
        executed = np.clip(raw, bounds[:, 0], bounds[:, 1])
        return executed, int(np.count_nonzero(raw != executed))


def generate_candidates(n=9, radius=.01, seed=110605):
    if n < 3 or n > 33 or not 0 < radius <= .05:
        raise ValueError('Use 3..33 candidates and a residual radius in (0, .05]')
    rng = np.random.default_rng(seed)
    out = [Candidate('baseline', (0.,) * 7)]
    # Antithetic directions; preserve the baseline gripper command (index 6).
    while len(out) < n:
        direction = rng.normal(size=7)
        direction[6] = 0
        direction *= radius / np.max(np.abs(direction))
        for sign in (1, -1):
            if len(out) < n:
                out.append(Candidate(f'candidate_{len(out):02d}', tuple(sign * direction)))
    return out


@dataclass
class Episode:
    source: str
    evidence_kind: str
    split: str
    seed: int
    initial_hash: str
    candidate: str
    states: np.ndarray
    actions: np.ndarray
    height: np.ndarray
    contact: np.ndarray
    hits: np.ndarray
    terminated: np.ndarray
    clipped_elements: int = 0

    @property
    def success(self):
        # Ignore reset-only success; initial successes are rejected by the runner.
        return bool(np.any(self.hits[1:]))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, states=self.states, actions=self.actions,
                            next_states=self.states[1:], height=self.height,
                            contact=self.contact, success_at_step=self.hits,
                            terminated=self.terminated,
                            truncated=np.array(not self.terminated[-1]),
                            reward=self.hits[1:].astype(float),
                            metadata=np.array(json.dumps(self.metadata())))

    def metadata(self):
        return {k: getattr(self, k) for k in
                ('source', 'evidence_kind', 'split', 'seed', 'initial_hash',
                 'candidate', 'clipped_elements')} | {'success': self.success}


def rollout(env, controller, initial, candidate, bounds, horizon, source, split, evidence_kind):
    # Models receive only an initial observation, never the simulator or its snapshot.
    if source == 'world_model':
        obs0 = copy.deepcopy(initial.observation)
        obs0.official_success = None
        obs = env.reset(obs0)
    else:
        obs = env.reset(copy.deepcopy(initial))
        if not np.allclose(obs.policy_state, initial.observation.policy_state, atol=1e-8, rtol=0):
            raise ValueError('Simulator reset changed the controller initial state')
        if not np.allclose([obs.height, obs.contact],
                           [initial.observation.height, initial.observation.contact], atol=1e-8, rtol=0):
            raise ValueError('Simulator reset changed task state')
    controller.reset()
    observations, actions, clipped = [copy.deepcopy(obs)], [], 0
    obs.validate(source)
    for _ in range(horizon):
        packet, count = candidate.apply(controller.packet(copy.deepcopy(obs)), bounds)
        obs = env.step(packet.copy())
        obs.validate(source)
        observations.append(copy.deepcopy(obs))
        actions.append(packet.copy())
        clipped += count
        if obs.terminated:
            break
    if source == 'simulator':
        hits = [bool(o.official_success) for o in observations]
    else:
        hits = [o.height > .925 and o.contact >= .5 for o in observations]
    return Episode(source, evidence_kind, split, initial.seed, initial.digest, candidate.name,
                   np.stack([o.policy_state for o in observations]), np.stack(actions),
                   np.array([o.height for o in observations]),
                   np.array([o.contact for o in observations]), np.array(hits),
                   np.array([o.terminated for o in observations]), clipped)


def confusion(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError('Expected nonempty paired labels')
    tp, fn = int((actual & predicted).sum()), int((actual & ~predicted).sum())
    fp, tn = int((~actual & predicted).sum()), int((~actual & ~predicted).sum())
    return dict(n=len(actual), tp=tp, fn=fn, fp=fp, tn=tn,
                simulator_successes=int(actual.sum()), predicted_successes=int(predicted.sum()),
                agreement=(tp + tn) / len(actual),
                false_success_among_failures=fp / (fp + tn) if fp + tn else None,
                failure_recall=tn / (fp + tn) if fp + tn else None,
                false_success_among_predictions=fp / (tp + fp) if tp + fp else None)


def paired_difference(before, after, seed=705, repeats=4000):
    a, b = np.asarray(before, int), np.asarray(after, int)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError('Expected nonempty paired binary outcomes')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Outcomes must be binary')
    d = b - a
    wins, losses = int((d == 1).sum()), int((d == -1).sum())
    n_discordant = wins + losses
    p = min(1., 2 * sum(math.comb(n_discordant, k) for k in range(min(wins, losses) + 1)) / 2**n_discordant)
    samples = np.random.default_rng(seed).choice(d, size=(repeats, len(d))).mean(axis=1)
    return dict(n=len(a), before_successes=int(a.sum()), after_successes=int(b.sum()),
                before_rate=float(a.mean()), after_rate=float(b.mean()),
                difference=float(d.mean()), wins=wins, losses=losses,
                both_success=int(((a == 1) & (b == 1)).sum()),
                both_failure=int(((a == 0) & (b == 0)).sum()),
                paired_bootstrap_95ci=np.quantile(samples, [.025, .975]).tolist(),
                exact_mcnemar_p=p,
                interval_note='Percentile paired-state bootstrap; descriptive for small n. Degenerate samples can give zero-width intervals.')


def ranking_metrics(rows, score_key):
    # Conditional ordering accuracy: simulator ties excluded; model ties get .5.
    groups = {}
    for r in rows:
        groups.setdefault(r['seed'], []).append(r)
    values, zero_variance = [], 0
    for g in groups.values():
        zero_variance += int(np.std([r[score_key] for r in g]) < 1e-12)
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                truth = int(a['actual']) - int(b['actual'])
                if truth:
                    delta = a[score_key] - b[score_key]
                    values.append(.5 if abs(delta) < 1e-12 else float(delta * truth > 0))
    return dict(informative_pairs=len(values),
                pairwise_accuracy=float(np.mean(values)) if values else None,
                zero_variance_groups=zero_variance, groups=len(groups))


def features(episode, candidate):
    # Task-conditioned reward correction, not a replacement dynamics model.
    h, c = episode.height[1:], episode.contact[1:]
    simultaneous_margin = np.max(np.minimum((h - .925) / .05, c - .5))
    return np.array([float(episode.success), (h.max() - .925) / .05,
                     (h[-1] - .925) / .05, c.max(), c[-1], simultaneous_margin,
                     *candidate.residual_fraction], dtype=float)


@dataclass
class RewardCorrection:
    """Ridge-regularized logistic head trained ONLY on collection simulator labels."""
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, rows, steps=600, lr=.08, ridge=.02):
        if not rows or any(r['split'] != 'collect' or r['label_source'] != 'simulator' for r in rows):
            raise ValueError('Only simulator collection labels may train reward correction')
        x = np.array([r['features'] for r in rows])
        y = np.array([r['actual'] for r in rows], dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError('Calibration needs both outcomes; retain the uncorrected model')
        mean = x.mean(0)
        std = x.std(0)
        # Normalize each physical scale separately, including small residuals.
        scale = np.where(std > 1e-8, std, 1.)
        z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        weights = np.zeros(z.shape[1])
        for _ in range(steps):
            pred = 1 / (1 + np.exp(-np.clip(z @ weights, -40, 40)))
            regularizer = ridge * weights.copy()
            regularizer[0] = 0
            weights -= lr * (z.T @ (pred - y) / len(y) + regularizer)
        return cls(mean, scale, weights)

    def predict(self, x):
        z = np.column_stack([np.ones(len(x)), (np.asarray(x) - self.mean) / self.scale])
        return 1 / (1 + np.exp(-np.clip(z @ self.weights, -40, 40)))

    def save(self, path):
        write_json(path, {k: v.tolist() for k, v in asdict(self).items()})

    @classmethod
    def load(cls, path):
        return cls(**{k: np.asarray(v) for k, v in json.loads(Path(path).read_text()).items()})


@dataclass
class Config:
    # 112 packets = 22.4 s: both branches stop at the same packet boundary.
    horizon_packets: int = 112
    candidates: int = 5
    residual_radius: float = .01
    candidate_seed: int = 110605
    verify_top_k: int = 2
    collect: list[int] = field(default_factory=lambda: list(range(11061000, 11061004)))
    validate: list[int] = field(default_factory=lambda: list(range(11062000, 11062004)))
    search: list[int] = field(default_factory=lambda: list(range(11063000, 11063004)))
    confirm: list[int] = field(default_factory=lambda: list(range(11064000, 11064004)))
    test: list[int] = field(default_factory=lambda: list(range(11065000, 11065010)))

    def check(self):
        all_seeds = []
        for name in ('collect', 'validate', 'search', 'confirm', 'test'):
            seeds = getattr(self, name)
            if not seeds or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError(f'Invalid {name} seeds')
            all_seeds.extend(seeds)
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError('Splits overlap or contain duplicates')
        if set(all_seeds) & set(range(10290200, 10290300)):
            raise ValueError('Stage 4 historical test seeds are not fresh held-out states')
        if not 1 <= self.horizon_packets <= 112 or not 1 <= self.verify_top_k < self.candidates:
            raise ValueError('Invalid bounded experiment budget')


def run_cycle(simulator, world_model, controller, output, config=None, candidates=None, provenance=None):
    config = config or Config()
    config.check()
    output = Path(output)
    # Never overwrite a previous run, including a failed or partially completed run.
    output.mkdir(parents=True, exist_ok=False)
    candidates = candidates or generate_candidates(config.candidates, config.residual_radius, config.candidate_seed)
    if candidates[0].name != 'baseline' or any(candidates[0].residual_fraction):
        raise ValueError('First candidate must be the unchanged baseline')
    if len(candidates) != config.candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError('Candidate count/names do not match the configuration')
    if any(len(c.residual_fraction) != 7 or not np.isfinite(c.residual_fraction).all()
           or np.max(np.abs(c.residual_fraction)) > config.residual_radius + 1e-12
           or c.residual_fraction[6] != 0 for c in candidates):
        raise ValueError('Candidate residual violates the frozen search space')
    bounds = np.asarray(simulator.bounds)
    if bounds.shape != (7, 2) or not np.isfinite(bounds).all() or not np.all(bounds[:, 1] > bounds[:, 0]):
        raise ValueError('Invalid actuator bounds')
    evidence = simulator.evidence_kind
    if evidence not in ('mujoco', 'synthetic_test'):
        raise ValueError('Unknown evidence kind')
    write_json(output / 'run.json', {'status': 'RUNNING', 'evidence_kind': evidence,
               'config': asdict(config), 'candidates': [asdict(c) for c in candidates],
               'provenance': provenance or {}, 'policy_optimizer': 'bounded_candidate_search',
               'model_update': 'reward_correction_head; latent dynamics frozen'})
    baseline = candidates[0]
    starts, episodes, records = {}, {}, []
    initial_hashes = {}

    def get_start(split, seed):
        if seed not in starts:
            start = simulator.initial(seed)
            start.observation.validate('simulator')
            if start.observation.official_success or start.observation.terminated:
                raise ValueError('Invalid initial terminal/successful state')
            # Do not permit different seed aliases for one physical initial state.
            if start.digest in initial_hashes:
                raise ValueError('Repeated physical initial state across seeds')
            initial_hashes[start.digest] = seed
            starts[seed] = start
            path = output / 'initial_states' / f'{seed}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **start.snapshot)
            write_json(path.with_suffix('.json'), {'seed': seed, 'split': split, 'sha256': start.digest,
                       'policy_state': start.observation.policy_state.tolist(),
                       'height': start.observation.height, 'contact': start.observation.contact})
        return starts[seed]

    def evaluate(split, seed, candidate, source):
        key = split, seed, candidate.name, source
        if key not in episodes:
            start = get_start(split, seed)
            env = simulator if source == 'simulator' else world_model
            episode = rollout(env, controller, start, candidate, bounds, config.horizon_packets,
                              source, split, evidence)
            episode.save(output / 'trajectories' / split / f'{seed}_{candidate.name}_{source}.npz')
            episodes[key] = episode
        return episodes[key]

    def paired_rows(split, selected):
        rows = []
        for seed in getattr(config, split):
            for candidate in selected:
                prediction = evaluate(split, seed, candidate, 'world_model')
                actual = evaluate(split, seed, candidate, 'simulator')
                row = dict(split=split, seed=seed, candidate=candidate.name,
                           initial_hash=actual.initial_hash, label_source='simulator',
                           evidence_kind=evidence, actual=int(actual.success),
                           predicted=int(prediction.success),
                           prediction_error=int(prediction.success) - int(actual.success),
                           false_success=bool(prediction.success and not actual.success),
                           missed_success=bool(actual.success and not prediction.success),
                           features=features(prediction, candidate).tolist())
                records.append(row)
                rows.append(row)
        return rows

    collect = paired_rows('collect', candidates)
    validation = paired_rows('validate', candidates)
    head = None
    update_reason = 'No outcome diversity in collection; original predictions retained.'
    correction_report = {'accepted': False}
    if len({r['actual'] for r in collect}) == 2:
        fitted = RewardCorrection.fit(collect)
        fitted.save(output / 'reward_correction_candidate.json')
        y = np.array([r['actual'] for r in validation])
        raw = np.array([r['predicted'] for r in validation])
        corrected = fitted.predict([r['features'] for r in validation])
        old_cm, new_cm = confusion(y, raw), confusion(y, corrected >= .5)
        old_brier, new_brier = float(np.mean((raw - y)**2)), float(np.mean((corrected - y)**2))
        accept = new_brier < old_brier - 1e-12 and new_cm['fp'] <= old_cm['fp'] and new_cm['fn'] <= old_cm['fn']
        correction_report = dict(accepted=accept, before=old_cm, after=new_cm,
                                 before_brier=old_brier, after_brier=new_brier,
                                 reason='Brier must strictly decrease; neither false successes nor missed successes may increase on validation.')
        update_reason = 'Validation gate passed.' if accept else 'Validation gate failed; original predictions retained.'
        if accept:
            head = fitted
            head.save(output / 'reward_correction_selected.json')

    def score_rows(rows):
        scores = head.predict([r['features'] for r in rows]) if head is not None else [r['predicted'] for r in rows]
        for r, score in zip(rows, scores):
            r['corrected_score'] = float(score)
            if 'actual' in r:
                r['corrected_score_error'] = float(score - r['actual'])
        return rows

    score_rows(collect)
    score_rows(validation)
    search = []
    for seed in config.search:
        for candidate in candidates:
            prediction = evaluate('search', seed, candidate, 'world_model')
            search.append(dict(split='search', seed=seed, candidate=candidate.name,
                               predicted=int(prediction.success), features=features(prediction, candidate).tolist()))
    score_rows(search)
    mean_scores = {c.name: float(np.mean([r['corrected_score'] for r in search if r['candidate'] == c.name])) for c in candidates}
    # Keep baseline in the shortlist regardless of its imagined score.
    challengers = sorted(candidates[1:], key=lambda c: (-mean_scores[c.name], c.name))[:config.verify_top_k]
    confirmation = score_rows(paired_rows('confirm', [baseline, *challengers]))
    actual_scores = {c.name: sum(r['actual'] for r in confirmation if r['candidate'] == c.name)
                     for c in [baseline, *challengers]}
    selected = baseline
    for candidate in challengers:
        if actual_scores[candidate.name] > actual_scores[selected.name]:
            selected = candidate
    # Freeze selection BEFORE obtaining a single final-test observation/outcome.
    write_json(output / 'selected_policy.json', {'candidate': asdict(selected),
               'confirmation_successes': actual_scores, 'selection_source': 'simulator_confirm',
               'reason': 'Strictly greater confirmation success count; baseline wins ties.'})
    write_json(output / 'search.json', {'mean_predicted_scores': mean_scores, 'rows': search})
    final_candidates = [baseline] if selected == baseline else [baseline, selected]
    final = score_rows(paired_rows('test', final_candidates))
    before = [next(r['actual'] for r in final if r['seed'] == seed and r['candidate'] == baseline.name) for seed in config.test]
    after = [next(r['actual'] for r in final if r['seed'] == seed and r['candidate'] == selected.name) for seed in config.test]
    errors = {}
    for candidate in final_candidates:
        rows = [r for r in final if r['candidate'] == candidate.name]
        errors[candidate.name] = {
            'raw': confusion([r['actual'] for r in rows], [r['predicted'] for r in rows]),
            'corrected': confusion([r['actual'] for r in rows], [r['corrected_score'] >= .5 for r in rows])}
    report = dict(status='COMPLETE', evidence_kind=evidence, selected_policy=selected.name,
                  policy_changed=selected != baseline, model_correction=correction_report,
                  correction_reason=update_reason, paired_policy_evaluation=paired_difference(before, after),
                  heldout_model_errors=errors, collection_ranking=ranking_metrics(collect, 'predicted'),
                  validation_ranking_before=ranking_metrics(validation, 'predicted'),
                  validation_ranking_after=ranking_metrics(validation, 'corrected_score'),
                  actual_simulator_episodes=sum(e.source == 'simulator' for e in episodes.values()),
                  imagined_episodes=sum(e.source == 'world_model' for e in episodes.values()),
                  model_dynamics_updated=False, grpo_performed=False)
    write_json(output / 'pairs.json', records)
    write_json(output / 'result.json', report)
    manifest = json.loads((output / 'run.json').read_text())
    manifest['status'] = 'COMPLETE'
    write_json(output / 'run.json', manifest)
    return report
