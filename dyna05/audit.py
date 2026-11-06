"""Historical evidence audit; never reports these labels as new executions."""
import json
from pathlib import Path
import numpy as np
from .core import confusion
from .assets import file_sha256


def audit_stage4(root):
    folder = Path(root) / 'media/stage4'
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, expected in manifest.items():
        if file_sha256(folder / name) != expected:
            raise ValueError(f'Stage 4 evidence changed: {name}')
    summary = json.loads((folder / 'summary.json').read_text())
    rows = summary['rows']
    if sorted(r['seed'] for r in rows) != list(range(10290200, 10290300)):
        raise ValueError('Historical seed table is incomplete')
    metrics = confusion([r['simulator'] for r in rows], [r['world_model'] for r in rows])
    traces = []
    for seed in summary['cases'].values():
        with np.load(folder / f'{seed}_trace.npz', allow_pickle=False) as trace:
            if trace['a'].shape[1:] != (10, 7) or trace['z'].shape[1] != 128:
                raise ValueError('Unexpected archived trace schema')
            traces.append(dict(seed=seed, model_steps=len(trace['a']), latent_dim=128))
    return dict(evidence_kind='historical_stage4_records', new_simulator_runs=0,
                checked_files=len(manifest), metrics=metrics, traces=traces,
                policy_sha256=summary['policy_sha256'])
