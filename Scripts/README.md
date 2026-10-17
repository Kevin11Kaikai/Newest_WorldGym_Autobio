# Reusing the existing ACT reproduction core

These files are copied from the earlier project's portable implementation. The top-level entries are preparation, ACT SFT and evaluation; their transitive runtime imports are preserved unchanged. Some shared runtime modules retain optional SmolVLA/reward functionality because the reused evaluator imports them. This focused release does not expose an RL training campaign.

## CPU record verification

From the repository root:

    python -m pip install -r requirements.txt
    python tools/verify_notebooks.py

This recounts all twenty Stage 2 records and the Stage 1 outcome table. No simulator, GPU or checkpoint is needed.

## GPU preparation

Use Linux/WSL2 and CUDA. AutoBio includes a Linux MuJoCo plugin. The inherited Dockerfile describes the environment; a new from-scratch build has not been validated for this release.

    docker build -f Scripts/Dockerfile -t worldgym-autobio .
    docker run --gpus all --shm-size=8g --rm -it -v "$PWD:/repo" worldgym-autobio
    python Scripts/prepare.py upstreams
    python Scripts/prepare.py public-models --profile act_native
    python Scripts/prepare.py import-task /path/to/worldgym-autobio-task-assets
    python Scripts/prepare.py check --profile act_native --full-hash

**The task bundle is local and has not been published.** The path above is a required input, not a download URL. Public upstream weights are obtained from their original sources, with any required authorization. See configs/assets.json for pinned assets. An ignored assets.local.json can map existing asset names to paths; expected hashes are still checked.

## ACT continuation and evaluation

    # Later 40-demo continuation from the saved ACT baseline; NOT original 80-demo training.
    python Scripts/prepare.py data --download
    python Scripts/prepare.py check --profile act_sft --full-hash
    python Scripts/sft_act.py --run runs/act40 --condition 40 --seed 5101

    # A full ACT pretrained directory is required for each policy.
    python Scripts/evaluate.py --run runs/act_physical --policy act --checkpoint /path/to/act/pretrained --environment mujoco
    python Scripts/evaluate.py --run runs/act_native --policy act --checkpoint /path/to/act/pretrained --environment worldgym

The original Stage 1 10/80-demo rescue is archived evidence; sft_act.py supports the later 10/20/40-demo continuation arms, not a reconstruction of that original training campaign. Do not label its output as the notebook's historical 31/50 or 50/50.

The default evaluation configuration comes from a later portable runtime and uses its own scene lists. These commands rerun the model/physical workflows; they do **not** by themselves reproduce the historical ten-seed Native diagnostic table or its posthoc action replay. That table is verified from the supplied original records. Native output has no official physical success label; the archived right-hand column comes from a separate MuJoCo replay of saved native-loop controls.

--help does not allocate CUDA. --smoke runs a small engineering check, not a performance experiment. Outputs belong in new run directories; resume checks input and source hashes.

SOURCE_PROVENANCE.json and VALIDATION.json are unchanged records from the earlier portable package. They describe earlier tests, not newly executed GPU tests in this release. This release checks script compilation/help and notebook execution; it does not rerun expensive training, download checkpoints or claim a fresh full GPU reproduction.
