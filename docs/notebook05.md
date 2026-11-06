# Notebook 05: GRPO and latent-dynamics Dyna round

The existing `05_dyna_style_closed_loop.ipynb` now contains the training path.
No sixth notebook was created. `dyna05/training.py:run_training` orchestrates
actual PyTorch GRPO, physical data collection, latent transition fine-tuning,
equal-compute policy continuation, and paired simulator evaluation.

CPU PyTorch 2.6.0 and MuJoCo 3.3.0 are installed in the implementation workspace.
The original Stage 4 model/controller weights, source, normalizers and AutoBio
assets are still absent. Therefore there is no new AutoBio success-rate result.
Optimizer and full-pipeline tests use explicitly labelled synthetic fixtures;
the real CLI refuses these fixtures.

## Reproduce the available checks

```bash
python -m pip install -r requirements-notebook05-training.txt
python -m pytest tests -q
python tools/verify_notebooks.py
python -m dyna05.cli preflight --assets configs/notebook05_assets.example.json
```

Preflight returns exit code 2 until the original bundle is supplied. When the
host blocks kernel sockets, use `AUTOBIO_NOTEBOOK_EXECUTOR=inprocess python
tools/verify_notebooks.py`; this executes the actual cells in fresh IPython
processes. The normal CI path uses fresh Jupyter kernels.

## One bounded round

1. **Policy initialization:** keep the original state student and lift rule
   frozen (π0). Add a state-conditioned MLP residual, initialized at zero mean
   with fixed Gaussian standard deviation 0.5 (π_init). The untrained stochastic
   policy is a separate evaluation arm because noise/boundary transforms can
   change behavior before learning.
2. **GRPO A:** sample four trajectories per training initial state. Use binary
   imagined episode success and group-relative advantages to update the residual
   MLP for three iterations, two optimizer epochs each. The original model is T0.
3. **Physical collection:** run π0 and π1 in MuJoCo on four `wm_fit` states and
   four separate `wm_validate` states. Save actual state/action/next-state data.
4. **Latent fit:** 40 Adam steps with three-packet recursive windows; update
   explicitly named original transition/readout parameters and freeze encoder
   and normalization. Loss combines latent MSE, decoded-state normalized MSE,
   height MSE scaled by 0.05 m, and contact BCE. These are explicit pilot loss
   choices, not a claim to reproduce an undisclosed original training loss.
5. **Model selection:** accept only if held-out one-step loss decreases,
   open-loop state NMSE/height MSE/contact Brier do not increase, and neither
   episode false successes nor missed successes increases. Validation replays
   the same physical action sequences under T0 and candidate T1. Retain rejected
   weights and metrics; use T0 for the next stage if rejected.
6. **GRPO B and matched control:** start twice from the identical π1 checkpoint,
   including optimizer, reference and RNG. One arm trains in selected T1 and
   the other in frozen T0 for the same three iterations. The reference remains
   π_init. This controls for extra policy training.
7. **Physical confirmation:** evaluate π0, π_init, π1, π2 and the frozen-T0
   continuation on four identical starts. Among π1/π2/control, only strictly
   greater true success count can replace π0; ties retain the incumbent.
8. **Final test:** save selection and all weights before first accessing ten
   held-out physical starts. Evaluate all five policies from identical snapshots
   and with common policy-noise seeds. For each, also compare T0/selected-T1
   recursive predictions against those actual labels. Never select on this test.

The fixed budget is **86 physical episodes**, **144 GRPO imagined episodes**,
and **100 final prediction episodes**. Each episode has at most 112 packets
(22.4 seconds). Reset/bridge checks add no physical actions. Zero-variance GRPO
batches skip optimizer updates and report zero; they do not trigger invented
shaping rewards or extra training. A completed round need not improve anything.

Configuration is `configs/notebook05_training.json`. Seeds are disjoint across
GRPO, model fitting, model validation, confirmation and final test. Historical
Stage 4 test seeds 10290200–10290299 and supplied original training/dev seeds
are excluded. Small final samples yield descriptive paired evidence, not a
performance guarantee. We save paired wins/losses, bootstrap intervals and exact
McNemar tests; degenerate bootstrap intervals do not establish certainty.

## Correct probability for the executed action

The original controller still consumes its 50 x 7 chunk and emits a physical
10 x 7 packet. Six Gaussian residuals are sampled once per packet and shared by
its ten commands; the gripper remains unchanged. For each arm joint, intersect
all ten commands' feasible residual intervals with ±1% actuator range. Map the
Gaussian through tanh into that interval. No clipping occurs after sampling;
an out-of-bounds original command is rejected rather than silently altered.

Log probability includes the tanh and interval Jacobians and sums over active
joint dimensions. Zero-width dimensions are deterministic and excluded. The
PPO-style ratio is the **joint packet probability**, not seven separate token
ratios. Per-episode means mask padded packets before averaging episodes.
Population-standard-deviation group advantages become exactly zero for constant
rewards. Old log probabilities are frozen at collection. Analytic Gaussian KL
is exact under the shared bijective transform and uses a frozen reference.

`grpo.py` saves sampled Gaussian variables, original packets, feasible intervals,
old log probabilities, rollout source, model rewards, optimizer and RNG state.
`GRPOTrainer.load` resumes the optimizer and sampling stream; exact continuation
is tested. The full multi-stage runner does not yet resume midway after a crash.
Use a new output directory; partial files remain available for diagnosis.

## Frozen physical protocol

- π0 is `dense50_10000.pt` plus the original lift rule, not visual ACT fed latent
  vectors. Checkpoint SHA256 is
  `9e6fca7e3bb15b4dee2dcb8ce15b68d324370bd6b8eddcbe9da5886a31f0f2a1`.
- One packet = 10 commands x 10 physics steps = 0.2 s. Both loops use 112 packets.
  Historical failures ended at 22.5/22.6 s; these new runs do not claim exact
  numerical reproduction of historical 90/100.
- Success is any post-packet observation with height > 0.925 m and simultaneous
  contact. The physical label is `task.check()`; model contact threshold is 0.5.
- `mjSTATE_INTEGRATION` and initial observation are saved; simulator reset
  equality is checked. Imagined rollouts receive only the starting observation,
  without the official label, simulator object, or future physical states.

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

## Additional trainable bridge contract

The same `build(resources, schema)` dictionary must additionally return
`trainable_model`, a wrapper around the **loaded original Stage 4 torch module**.
Do not instantiate a replacement architecture. The minimal implementation is
CPU-only and deterministic; move the original loaded weights and normalizer
buffers to CPU and retain their original floating-point precision.

| Wrapper member | Exact requirement |
|---|---|
| `encode(state)` | Differentiable mapping from `[B, state_dim]` original controller-state representation to `[B, latent_dim]` |
| `transition(z, packet)` | Differentiable next latent; packet is `[B,10,7]` in **physical** units; original action normalization belongs inside the wrapper |
| `decode(z)` | Dictionary: `policy_state` `[B,state_dim]`, `height` `[B]` in meters, `contact_logits` `[B]`; reproduce original readouts |
| `state_mean`, `state_scale` | Registered frozen buffers for that same decoded state representation; scale positive; identity only if that is genuinely the original normalized representation |
| `dynamics_parameter_names` | Nonempty exact names from `named_parameters()` of the original transition weights to update |
| `readout_parameter_names` | Disjoint names of optional original decoder/reward readout weights to update |

The original recurrent model and the trainable wrapper must agree on three
consecutive baseline/perturbed packets before training proceeds. It also checks
repeatable resets and action dependence. These checks expose interface mistakes;
they do not prove that model rewards rank real policies correctly.

The loss treats true next encodings as detached targets and recursively feeds
predicted latents through the training window. It never trains on archived
imagined `z` traces. The optimizer must change at least one named transition
parameter; a head-only or no-op update raises an error. Frozen tensors must stay
identical. Independent multi-step and success-judgment validation can reject a
fit even when its training/prediction loss falls.

## Execute with the original assets

```bash
python -m dyna05.cli preflight --assets configs/notebook05_assets.local.json
python -m dyna05.cli train \
  --assets configs/notebook05_assets.local.json \
  --config configs/notebook05_training.json \
  --output runs/notebook05_grpo_round1
```

The notebook's `RUN_REAL` cell invokes this same command. No synthetic fallback
exists. `result.json` with `status=COMPLETE` and `evidence_kind=mujoco` records a
completed physical round. Check `grpo_optimizer_steps` and the model-selection
flags separately; all-equal rewards or rejected model updates remain visible.

## Saved evidence and interpretation

| Output | Meaning |
|---|---|
| `protocol.json`, `bridge_check.json` | Frozen configuration, budget, package/source/asset hashes, wrapper equivalence |
| `initials/<split>/*.npz` + JSON | Physical integration snapshots, initial observations, seed and snapshot fingerprint |
| `grpo_a`, `grpo_b`, `frozen_t0_control` | Actual imagined trajectories, decision likelihood traces, optimizer logs and resumable policy checkpoints |
| `episodes/<split>/<policy>/<source>/*.npz` | Physical/model states, executed actions, true or predicted reward, terminal/truncation and evidence source |
| `latent_fit/candidate.pt`, `selected.pt`, `validation.json` | Candidate dynamics + optimizer/RNG, selected model weights, independent before/after errors |
| `frozen_selection.json`, `*_frozen.pt` | Independent confirmation decision and policy weights saved before final test access |
| `final_pairs.json` | Per-policy/per-state actual label plus T0 and selected-T1 predictions |
| `result.json` | Baseline comparisons, π_init→π1 learning comparison, equal-compute control, confusion matrices and policy ranking |

`model_update_vs_extra_training` compares π2 with the frozen-T0 continuation;
`learning_vs_initial` separates learning from adding noise; `policy_comparisons`
compares each arm with π0. `selected_policy_comparison` evaluates the policy
chosen on confirmation, without test-driven promotion. Model errors are both
open-loop on held-out physical actions (validation) and closed-loop on the same
policies/starts (final test). If T1 was rejected, final `T1_selected` is T0 and
is labelled accordingly. Report negative results without changing the protocol.

## Earlier finite-candidate pilot

`dyna05/core.py:run_cycle` and `configs/notebook05.json` remain available through
`python -m dyna05.cli run`. That earlier route uses fixed residual candidates
and a success-reward correction head; its result explicitly says no GRPO/latent
update. The notebook now uses **`train`**, not that earlier route.

## Sources

- [Stage 4 notebook](../04_latent_tube_policy_evaluation.ipynb) and [paired labels](../media/stage4/summary.json).
- [World-Gymnast official code](https://github.com/world-gymnast/world-gymnast): inspiration for model-assisted policy optimization and iterative model updates; this pilot uses the existing AutoBio latent interface, not its visual VLA training implementation.
- [AutoBio pickup source](https://github.com/autobio-bench/AutoBio/blob/08504213356cbfe9a82bcd33a43cd282945a45df/autobio/pickup_centrifuge_tube.py): physical task and height/contact rule.
