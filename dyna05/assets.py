"""Asset preflight and explicit loading of the private Stage 4 bridge."""
import hashlib
import importlib.util
import json
from pathlib import Path

POLICY_SHA256 = '9e6fca7e3bb15b4dee2dcb8ce15b68d324370bd6b8eddcbe9da5886a31f0f2a1'
REQUIRED = ('policy_checkpoint', 'world_model_checkpoint', 'state_action_schema',
            'training_seed_manifest', 'bridge_source')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preflight(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    blockers, resolved = [], {}
    for name in REQUIRED:
        item = manifest.get(name, {})
        if not item.get('path'):
            blockers.append(f'{name}: path not supplied')
            continue
        resource = Path(item['path'])
        resource = resource if resource.is_absolute() else path.parent / resource
        if not resource.is_file():
            blockers.append(f'{name}: file missing: {resource}')
            continue
        expected = item.get('sha256')
        if not expected or len(expected) != 64:
            blockers.append(f'{name}: SHA256 not supplied')
            continue
        actual = file_sha256(resource)
        if actual != expected:
            blockers.append(f'{name}: SHA256 mismatch')
            continue
        if name == 'policy_checkpoint' and actual != POLICY_SHA256:
            blockers.append('policy_checkpoint: does not match the frozen Stage 4 baseline')
            continue
        resolved[name] = str(resource.resolve())
    for dependency in ('torch', 'mujoco'):
        if importlib.util.find_spec(dependency) is None:
            blockers.append(f'dependency missing: {dependency}')
    if manifest.get('stage4_source_available') is not True:
        blockers.append('Original Stage 4 controller/model class definitions and normalizers are not supplied')
    return dict(status='BLOCKED' if blockers else 'READY_FOR_BRIDGE_CHECK',
                blockers=blockers, resolved=resolved, manifest=manifest)


def load_real_bundle(manifest_path, config):
    result = preflight(manifest_path)
    if result['blockers']:
        raise RuntimeError('\n'.join(result['blockers']))
    resources = result['resolved']
    # Original training/dev seeds must be exported, not inferred from checkpoint names.
    history = json.loads(Path(resources['training_seed_manifest']).read_text())
    if history.get('complete') is not True or not isinstance(history.get('used_seeds'), list):
        raise ValueError('A complete original training/development seed manifest is required')
    split_values = config.splits().values() if hasattr(config, 'splits') else (
        getattr(config, x) for x in ('collect', 'validate', 'search', 'confirm', 'test'))
    new_seeds = set(sum(split_values, []))
    if new_seeds & set(history['used_seeds']):
        raise ValueError('New experiment seeds overlap original training/development seeds')
    schema = json.loads(Path(resources['state_action_schema']).read_text())
    if schema.get('packet_shape') != [10, 7] or schema.get('packet_seconds') != .2:
        raise ValueError('Stage 4 action schema differs from the audited packet interface')
    spec = importlib.util.spec_from_file_location('autobio_stage4_private_bridge', resources['bridge_source'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = module.build(resources=resources, schema=schema)
    from .mujoco_backend import MuJoCoTube
    if not isinstance(bundle['simulator'], MuJoCoTube):
        raise TypeError('Real mode requires the MuJoCoTube physical adapter')
    return bundle, result
