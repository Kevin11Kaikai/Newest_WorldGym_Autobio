# Notebook 05: one simulator-validated Dyna-style iteration

The entry point is `05_dyna_style_closed_loop.ipynb`. CPU execution audits the
archived evidence, displays the frozen experiment, runs independent contract
tests, and checks the real-run prerequisites. `run_cycle` implements the actual
candidate / prediction / feedback / correction / selection / final-evaluation
workflow. Real execution is blocked until the original Stage 4 bundle is supplied.

## What is implemented

1. Create the baseline plus four deterministic, antithetic actuator-residual
   candidates. Each is the **same feedback controller plus a fixed residual**,
   bounded to 1% of each actuator's physical range. The gripper is unchanged.
   There is no random exploration during an episode. This is finite candidate
   search, not GRPO and not a neural-policy weight update.
2. Run each candidate in the original model and MuJoCo on collection starts.
   Each loop uses its own evolving observations and recomputes actions. Later
   actions can differ between the two loops. This is policy evaluation, not
   open-loop replay of model-selected actions.
3. Fit a small logistic **success-reward correction head** from model rollout
   features and actual collection labels. Features include predicted height,
   contact, their simultaneous margin, and candidate residuals. Dynamics remain
   frozen. Neither a calibration score nor an imagined height is a physical reward.
4. Validate the head on separate starts. Use it only if Brier score strictly
   decreases and neither false successes nor missed successes increases. Save
   rejected heads too. No outcome diversity means no correction update.
5. Rank candidates on search starts, using model predictions only. Send the top
   two challengers and the baseline to independent MuJoCo confirmation starts.
   Replace the baseline only for a strictly higher actual success count. Ties
   retain the baseline. This gate is a pilot selection rule, not a statistical
   safety guarantee.
6. Freeze `selected_policy.json`, then obtain final-test starts. Evaluate the
   baseline and selected policy on identical physical snapshots. Do not change
   the selected policy after observing this test. A negative result is retained.

The head improves the **task-outcome predictor attached to the world model**.
It does not improve the latent transition function, and it does not affect the
controller's imagined height/contact feedback. That narrower update is explicit
in every result (`model_dynamics_updated=false`, `grpo_performed=false`). The
November 4 harness's stochastic GRPO and latent fine-tuning stages remain future
extensions, requiring the original architecture and weights. The November 6
request prioritizes this bounded first loop and independently testable code.

## Reproduce the available checks

From the repository root:

```bash
python -m pip install -r requirements-notebook05.txt
python -m pytest tests/test_dyna05.py -q
python -m dyna05.cli audit
python tools/verify_notebooks.py
python -m dyna05.cli preflight --assets configs/notebook05_assets.example.json
```

The final command intentionally returns exit code **2** with named blockers.
Tests use explicitly labeled synthetic dynamics, stored only in pytest temporary
directories. Those numbers are not saved as AutoBio research results. The
notebook's CPU checks require no PyTorch, MuJoCo, or task checkpoints.

In a workspace that forbids local kernel sockets, execute the exact cells in
separate fresh IPython processes instead:

```bash
AUTOBIO_NOTEBOOK_EXECUTOR=inprocess python tools/verify_notebooks.py
```

The verification report records which executor ran; outputs are captured from
actual cell execution in either mode. This fallback does not validate Jupyter's
socket transport.

## Physical protocol and budget

| Item | Frozen definition |
|---|---|
| Original policy | `dense50_10000.pt`, SHA256 `9e6fca7e3bb15b4dee2dcb8ce15b68d324370bd6b8eddcbe9da5886a31f0f2a1` |
| Controller interface | Original state student and lift rule; the bridge emits the actually executed 10 x 7 packet |
| Original action proposal | 50 x 7, with original chunk consumption preserved by the bridge |
| Packet timing | 10 commands x 10 physics steps = 0.2 s |
| New common horizon | 112 packets = **22.4 s**, identical for both branches |
| Historical horizon nuance | Saved failed simulator case ends at 22.5 s; model case at 22.6 s. This pilot deliberately uses a shared full-packet horizon, so it does not claim exact reproduction of historical 90/100 |
| Success | Any post-packet observation with height > 0.925 m and simultaneous contact; simulator uses `task.check()`; model contact threshold >= 0.5 |
| Residual | Added after the original controller/lift rule, then clipped; executed commands and clipping counts are saved |
| Initial states | `mjSTATE_INTEGRATION` snapshot saved once per seed; restore and observation equality checked before each simulator rollout |
| Splits | 4 collection, 4 validation, 4 model-search, 4 simulator-confirmation, 10 final-test starts |
| Maximum rollouts | **72 simulator + 92 model** episodes, fewer if baseline is retained |
| Reward | Episode success indicator; no fabricated shaped reward or early stopping on model optimism |

Defaults are an engineering pilot, not paper hyperparameters. The ten final
starts give a descriptive small-sample comparison. Paired wins/losses, paired
bootstrap intervals and exact McNemar p-values are saved; zero-width bootstrap
intervals for degenerate samples do not establish certainty. Final test IDs must
also be checked against the unavailable original training/development manifest.

No reset is called during imagined rollout after initialization. The model
receives an `Observation` with its official label removed, never a simulator
object or future physical states. The policy consumes the model's own decoded
policy-state representation; images and latent vectors are not interchangeable.

## Assets needed from the original experiment machine

The archived paths locate the original work under
`worldgym_autobio_mvp/experiments/test_tube_wm_repair_v2` and
`test_tube_wm_latest_v1`. That machine is not mounted in this workspace.

Supply these files in an ignored `configs/notebook05_assets.local.json` copied
from the example manifest:

- The frozen state-student checkpoint with the verified policy hash above.
- The actual tube `repair_v2` checkpoint at step 8000, with its SHA256. The
  public repository does not provide this hash; it must not be invented.
- Original controller, encoder, recurrent dynamics/readout source and any
  normalizer/statistics files they load. The public 128-value latent traces are
  outputs, not checkpoints or training-state schemas.
- A state/action schema documenting feature order, units, normalization and
  packet timing; JSON must include `packet_shape: [10, 7]` and
  `packet_seconds: 0.2` for the automated check.
- A complete original training/development seed list, JSON
  `{"complete": true, "used_seeds": [...]}`. The historical Stage 4 test range
  10290200–10290299 is rejected separately.
- Working AutoBio checkout, its task assets and native plugins, compatible
  MuJoCo/PyTorch dependencies, and the original architecture bridge below.

The Stage 2 task-adapted video checkpoint and PickCube TD-MPC2 checkpoint cannot
substitute for these assets. Upstream AutoBio commit
`08504213356cbfe9a82bcd33a43cd282945a45df` has `pickup_centrifuge_tube.py`, while
its `task.create_task('pickup')` imports `pickup`; use the working original task
loader or explicitly resolve this alias in that checkout. The core does not
silently patch third-party sources.

## Exact bridge contract

`bridge_source` is a **user-supplied Python file**, hash checked before loading.
It adapts the original architecture; it is not included as a fake working loader.
Its `build(resources, schema)` function returns a dictionary containing:

| Key | Required object |
|---|---|
| `simulator` | `dyna05.mujoco_backend.MuJoCoTube(original_task, observe)`; `observe(task)` returns the original normalized/unnormalized controller state exactly as required by the original controller |
| `controller` | `reset()` clears chunk/lift state; `packet(Observation)` returns one **physical** finite `(10,7)` command packet using the original student and lift behavior |
| `world_model` | `reset(initial_observation)` initializes the original encoder/recurrent state; `step(executed_packet)` recursively advances the original latent model; both return `Observation(policy_state, height, contact)` with `official_success=None` |

`Observation.policy_state` must have the same documented semantic meaning in
both loops. `controller.packet` owns the original 50-step chunk consumption,
the height >= 0.8847 m lift trigger, contact cancellation, 50-control-step lock,
and shoulder/elbow lift commands (-0.0686/-0.0933 rad/s). Do not guess whether
the source applies them before or after actuator conversion; copy its actual
implementation and verify a baseline replay. The local source is required
because the teaching notebook does not disclose all these transformations.

The original PyTorch objects should be in evaluation mode for inference.
Record all original source revisions and dependency versions with the asset
bundle. `load_real_bundle` refuses a synthetic simulator in real mode.
Before physical actions, the CLI checks model reset repeatability, matching
controller-state dimensions, and detectable response to positive/negative
action perturbations on a collection start. These are interface smoke checks,
not proof that predicted rewards rank policies correctly.

After implementing this architecture adapter with the original source:

```bash
python -m dyna05.cli preflight --assets configs/notebook05_assets.local.json
python -m dyna05.cli run \
  --assets configs/notebook05_assets.local.json \
  --config configs/notebook05.json \
  --output runs/notebook05_round1
```

Real mode never falls back to test fixtures or archived labels. Each run needs
a new directory. Partial trajectory files remain after failure; this minimal
version does not implement mid-run resume, so use a new output directory after
fixing the failure. A run without `result.json` and `status=COMPLETE` is incomplete.

## Saved evidence

- `run.json`: config, candidates, asset/source hashes, provenance and run status.
- `initial_states/*.npz` + JSON: exact physical starts, identifiers and hashes.
- `trajectories/<split>/*.npz`: state/action/next-state alignment, height/contact,
  per-step success, termination, truncation, clipping and source labels.
- `pairs.json`: raw model calls, corrected scores, actual labels and errors.
- `reward_correction_candidate.json` and, only on acceptance, the selected head.
- `search.json`: predicted candidate rankings; `selected_policy.json`: actual
  confirmation scores and frozen selected residual policy.
- `result.json`: simulator policy changes, paired uncertainty, raw/corrected
  model confusion matrices on final states, ranking, counts and scope flags.

Latent dynamics fine-tuning can later consume the saved **simulator** transitions
using the original training loss. It needs its own independent validation and
matched extra-policy-training control. Do not relabel the correction head as
that experiment.

## References

- [Stage 4 notebook](../04_latent_tube_policy_evaluation.ipynb) and [paired labels](../media/stage4/summary.json).
- [World-Gymnast official code](https://github.com/world-gymnast/world-gymnast) and [paper/project](https://world-gymnast.github.io/): learned-video rollout policy training and iterative world-model updates. This pilot retains the feedback logic but uses a latent model, finite residual search and reward correction.
- [AutoBio pickup source](https://github.com/autobio-bench/AutoBio/blob/08504213356cbfe9a82bcd33a43cd282945a45df/autobio/pickup_centrifuge_tube.py): official height/contact success rule.
