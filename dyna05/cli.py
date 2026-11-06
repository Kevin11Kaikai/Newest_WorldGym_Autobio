"""python -m dyna05.cli audit | preflight | run"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import platform
import time
from .audit import audit_stage4
from .assets import preflight, load_real_bundle, file_sha256
from .core import Config, run_cycle, write_json, check_bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--root', default='.')
    pre = sub.add_parser('preflight')
    pre.add_argument('--assets', default='configs/notebook05_assets.example.json')
    run = sub.add_parser('run')
    run.add_argument('--assets', required=True)
    run.add_argument('--config', default='configs/notebook05.json')
    run.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(audit_stage4(args.root), indent=2))
        return 0
    if args.command == 'preflight':
        result = preflight(args.assets)
        print(json.dumps(result, indent=2))
        return 2 if result['blockers'] else 0
    config = Config(**json.loads(Path(args.config).read_text()))
    config.check()
    bundle, audit = load_real_bundle(args.assets, config)
    start = time.perf_counter()
    provenance = dict(python=platform.python_version(), platform=platform.platform(),
                      assets={key: {'sha256': file_sha256(value)} for key, value in audit['resolved'].items()},
                      sources={str(p): file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
                      config_sha256=file_sha256(args.config))
    try:
        provenance['bridge_check'] = check_bridge(bundle['simulator'], bundle['world_model'],
                                                  bundle['controller'], config.collect[0])
        result = run_cycle(bundle['simulator'], bundle['world_model'], bundle['controller'],
                           args.output, config, provenance=provenance)
        result['elapsed_seconds'] = time.perf_counter() - start
        write_json(Path(args.output) / 'result.json', result)
        print(json.dumps(result, indent=2))
    finally:
        bundle['simulator'].close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
