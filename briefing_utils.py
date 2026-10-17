"""Small CPU-only evidence readers and plots for the two stage lessons."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display

ROOT = Path(__file__).resolve().parent
TEAL, NAVY, ORANGE, GRAY = '#168b83', '#183b52', '#cc6b42', '#879ba8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.bbox': 'tight'})


def data(name):
    return json.loads((ROOT / 'evidence' / name).read_text(encoding='utf-8-sig'))

def table(rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    display(HTML(frame.to_html(index=False, escape=True, border=0)))

def verify_assets():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert len({row['path'] for row in manifest['files']}) == len(manifest['files'])
    for row in manifest['files']:
        assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    return len(manifest['files'])

def input_views():
    with np.load(ROOT / 'evidence/observation_example.npz') as sample:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
        for ax, key, title in zip(axes, ['front', 'wrist'], ['Front camera', 'Wrist camera']):
            ax.imshow(sample[key]); ax.set_title(title); ax.axis('off')
        fig.suptitle('Actual saved observation | later ACT40 demonstration', fontsize=11)
        fig.tight_layout(); plt.show()
        assert sample['state'].shape == (7,) and sample['action_chunk'].shape == (50, 7)
        print('Joint-state shape:', sample['state'].shape)
        print('Action-chunk shape:', sample['action_chunk'].shape)

def phase1_results():
    frame = pd.read_csv(ROOT / 'evidence/phase1_outcomes.csv')
    rows = []
    for condition in [10, 80]:
        group = frame[frame.condition.eq(condition)]
        rows.append([f'{condition}-demo ACT', f'{int(group.success.sum())}/{len(group)}'])
    table(rows, ['Policy', 'Successes on the shared 50 starts'])

def check_stage_evidence():
    p1 = pd.read_csv(ROOT / 'evidence/phase1_outcomes.csv')
    assert len(p1) == 100 and p1.valid.eq(1).all()
    one = p1.pivot(index='seed', columns='condition', values='success')
    assert one.shape == (50, 2) and not one.isna().any().any()
    assert one.sum().to_dict() == {10: 31, 80: 50}
    p2 = pd.read_csv(ROOT / 'evidence/stage2_native_outcomes.csv')
    assert len(p2) == 20 and not p2.duplicated(['model', 'seed']).any()
    assert p2.groupby('model').size().tolist() == [10, 10]
    assert p2.pivot(index='seed', columns='model', values='mujoco_success').notna().all().all()
    assert set(one.index).isdisjoint(set(p2.seed))
    records = data('stage2_source_records.json')['records']
    for row in p2.itertuples():
        rec = next(r for r in records if r['model'] == row.model and r['seed'] == row.seed)
        ref, native, replay = rec['reference_episode'], rec['native_score'], rec['replay_diagnostic']
        assert row.mujoco_success == int(ref['official_early_stop_success'])
        assert row.native_command_replay_success == int(replay['physical_replay_official_success'])
        assert row.native_command_replay_success == int(any(rec['replay_saved_success_flags']))
        assert native['checkpoint_sha256'] == ref['checkpoint_sha256'] == row.policy_sha256
        assert replay['commands_sha256'] == row.commands_sha256
        assert native['all_post_start_actions_generated_by_original_ACT']
        assert native['independent_learned_observation_loop']
        assert not any(native[k] for k in ['future_true_RGB_input','future_true_qpos_input','future_true_controls_input'])
        assert ref['valid'] and native['valid'] and replay['physical_replay_valid']
    for expected in data('stage2_native_summary.json')['policy_results']:
        part = p2[p2.policy.eq(expected['policy'])]
        assert len(part) == expected['scenes']
        assert int(part.mujoco_success.sum()) == expected['mujoco_success']
        assert int(part.native_command_replay_success.sum()) == expected['native_command_replay_success']
    return {'stage1_scenes': 50, 'stage2_development_scenes': 10, 'status': 'PASS_SAVED_RECORD_CHECKS', 'new_inference': False}

def native_evaluation_results():
    frame = pd.read_csv(ROOT / 'evidence/stage2_native_outcomes.csv')
    rows = []
    for name in ['10-demo ACT', '80-demo ACT']:
        part = frame[frame.policy.eq(name)]
        rows.append([name, f'{int(part.mujoco_success.sum())}/{len(part)}',
                     f'{int(part.native_command_replay_success.sum())}/{len(part)}'])
    table(rows, ['Frozen policy', 'MuJoCo observation loop', 'Native-loop commands replayed in MuJoCo'])
