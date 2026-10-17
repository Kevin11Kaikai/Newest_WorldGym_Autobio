"""Common entrypoint validation. Help works without importing Torch or allocating a GPU."""
import contextlib, json, os, shutil, sys, time
from pathlib import Path
from control import R,asset,catalog,config,read,write,sha,now

PROFILES={
 'act_sft':['lerobot','decoded','act_base','act_stats'],
 'act_mujoco':['lerobot','autobio'],
 'act_native':['lerobot','autobio','worldgym_source','worldgym_base','worldgym_parent','worldgym_task','action_stats','proprio','vae'],
 'sft':['smolvla','smolvlm','lerobot','decoded'],
 'mujoco':['smolvla','smolvlm','lerobot','autobio'],
 'native':['smolvla','smolvlm','lerobot','autobio','worldgym_source','worldgym_base','worldgym_parent','worldgym_task','action_stats','proprio','vae'],
 'rl':['smolvla','smolvlm','lerobot','autobio','worldgym_source','worldgym_base','worldgym_parent','worldgym_task','action_stats','proprio','vae','qwen7b','reward_adapter'],
 'mujoco_rl':['smolvla','smolvlm','lerobot','autobio','qwen7b','reward_adapter'],
}
def common(parser):
    parser.add_argument('--run',required=True,type=Path,help='New output directory; resume only with identical inputs and code.')
    parser.add_argument('--assets',type=Path,help='Optional local asset path overrides; never commit this file.')
    parser.add_argument('--config',type=Path,default=R/'Scripts/configs/reproduce.json')
    parser.add_argument('--max-hours',type=float,help='Optional wall-clock limit, persisted on first start and not reset.')
    parser.add_argument('--resume',action='store_true',help='Continue a matching run from verified completed checkpoints.')
def configure(args):
    os.environ['WORLDGYM_RUN']=str(args.run.resolve())
    os.environ['WORLDGYM_CONFIG']=str(args.config.resolve())
    if args.assets:os.environ['WORLDGYM_ASSETS']=str(args.assets.resolve())
    # control.H was imported by this lightweight CLI before parsing.
    import control
    control.H=args.run.resolve()
def resolve_checkpoint(value):
    if value in catalog():
        path=asset(value).resolve()
        for rel,entry in catalog()[value].get('files',{}).items():
            p=path if rel=='.' else path/rel
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
                raise ValueError('Named task asset is missing or has changed: '+str(p))
        return path
    return Path(value).resolve()
def autobio_patch(root,apply=False):
    """Verify exactly two edits against pinned Git blobs; tolerate CRLF/LF only."""
    import subprocess
    edits={
      'autobio/task.py':('from pickup import Pickup','from pickup_centrifuge_tube import Pickup'),
      'autobio/pickup_centrifuge_tube.py':("mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')","from pathlib import Path\nmujoco.mj_loadPluginLibrary(str(Path(__file__).resolve().parent / 'libmjlab.so.3.3.0'))"),
    }
    for rel,(old,new) in edits.items():
        baseline=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+rel]).decode('utf-8').replace('\r\n','\n')
        if baseline.count(old)!=1:raise ValueError('Pinned AutoBio patch context changed: '+rel)
        expected=baseline.replace(old,new);path=root/rel;actual=path.read_text(encoding='utf-8')
        if apply and actual==baseline:path.write_text(expected,encoding='utf-8',newline='\n');actual=expected
        if actual!=expected:raise ValueError('AutoBio source differs from the exact compatibility patch: '+rel)
def verify_assets(profile,full=False):
    import subprocess
    cachefile=R/'outputs/asset_verification_cache.json'
    cache=read(cachefile) if cachefile.exists() else {}
    findings=[]
    for key in PROFILES[profile]:
        spec=catalog()[key];root=asset(key)
        if not root.exists():raise FileNotFoundError(f'{key}: missing {root}. See Scripts/README.md; no download or CPU fallback is attempted.')
        if spec['kind']=='git':
            rev=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
            if rev!=spec['revision']:raise ValueError(f'{key}: unexpected upstream commit {rev}')
            if key=='autobio':
                autobio_patch(root)
        elif spec['kind']=='decoded_dataset':
            d=config()['data'];episodes=d['splits']['40']+d['validation_episodes']
            for ep in episodes:
                for suffix in ['numeric.npz','image.npy','wrist_image.npy']:
                    p=root/f'episode_{ep:06d}_{suffix}'
                    if not p.is_file():raise FileNotFoundError(p)
        else:
            for rel,entry in spec.get('files',{}).items():
                p=root if rel=='.' else root/rel
                if not p.is_file():raise FileNotFoundError(p)
                stamp=[p.stat().st_size,p.stat().st_mtime_ns,entry['sha256']]
                if stamp[0]!=entry['bytes']:raise ValueError(f'Wrong asset size: {p}')
                if full or cache.get(str(p))!=stamp:
                    if sha(p)!=entry['sha256']:raise ValueError(f'Asset SHA256 mismatch: {p}')
                    cache[str(p)]=stamp
        findings.append(dict(asset=key,path=str(root),status='VERIFIED'))
    write(cachefile,cache)
    return findings
def begin(args,profile,settings,stats=None):
    configure(args)
    # This is a compute workflow, so refuse silent CPU fallback.
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required. CPU evidence checks use tools/verify_notebooks.py.')
    verified=verify_assets(profile)
    if stats:
        stats=resolve_checkpoint(str(stats))
        if not stats.is_file():raise FileNotFoundError(stats)
        settings['normalization_sha256']=sha(stats)
    code={p.relative_to(R).as_posix():sha(p) for p in sorted((R/'Scripts').rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.patch') and '.local.' not in p.name}
    identity=dict(settings=settings,source_hashes=code,config_sha256=sha(args.config))
    h=args.run.resolve();mf=h/'RUN_MANIFEST.json'
    if mf.exists():
        old=read(mf)
        if not args.resume:raise FileExistsError(f'{h} exists; use a new directory or explicit --resume.')
        if old['identity']!=identity:raise ValueError('Cannot resume with changed source, configuration, checkpoint, or scene selection.')
    else:
        if h.exists() and any(h.iterdir()):raise FileExistsError(f'Refusing nonempty output directory: {h}')
        if args.resume:raise FileNotFoundError('No existing run to resume.')
        h.mkdir(parents=True,exist_ok=True);t=time.time()
        write(mf,dict(status='STARTED',utc=now(),identity=identity,started_epoch=t,
                      deadline_epoch=t+args.max_hours*3600 if args.max_hours else None,
                      assets=verified,device=torch.cuda.get_device_name(0),purpose=settings.get('purpose','REPRODUCTION')))
    if stats:
        target=h/'DATA.json'
        if target.exists():
            if sha(target)!=sha(stats):raise ValueError('Normalization changed during resume.')
        else:shutil.copyfile(stats,target)
    return config()
@contextlib.contextmanager
def gpu_lock():
    # Linux/WSL is required by AutoBio's native .so; both language repos may share assets.
    import fcntl
    p=Path('/tmp/worldgym-autobio-gpu.lock')
    with p.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another WorldGym reproduction process owns the GPU lock.')
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
def finish(run,result):
    m=read(run/'RUN_MANIFEST.json');m.update(status='COMPLETE',completed_utc=now(),wall_seconds=time.time()-m['started_epoch'])
    write(run/'RUN_MANIFEST.json',m);write(run/'RESULT.json',result)
    write(run/'RUN_STATE.json',dict(status='COMPLETE',utc=m['completed_utc'],wall_seconds=m['wall_seconds']))
    print(json.dumps(result,indent=2),flush=True)
