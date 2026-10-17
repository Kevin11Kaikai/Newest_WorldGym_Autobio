"""Portable paths, immutable run settings, and optional resumable time limit."""
import datetime, hashlib, json, os, time
from pathlib import Path

R=Path(__file__).resolve().parents[2]
H=Path(os.environ.get('WORLDGYM_RUN',str(R/'runs/unconfigured'))).resolve()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def append(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
def config():return read(os.environ.get('WORLDGYM_CONFIG',R/'Scripts/configs/reproduce.json'))
def catalog():return read(R/'Scripts/configs/assets.json')['assets']
def asset(name):
    overrides=read(os.environ['WORLDGYM_ASSETS']) if os.environ.get('WORLDGYM_ASSETS') else {}
    value=overrides.get(name,catalog()[name]['path']);path=Path(value)
    return path if path.is_absolute() else R/path
def state(stage,**kw):
    value=dict(status='RUNNING',stage=stage,utc=now(),pid=os.getpid());value.update(kw)
    write(H/'RUN_STATE.json',value)
def log(message):
    with (H/'WORK_LOG.md').open('a',encoding='utf-8') as f:f.write('\n- '+now()+' '+message+'\n')
def budget(reserve=0):
    # Historical reserve arguments belong to the old scheduler, not a new run.
    # This run's optional absolute deadline is established once in cli.begin.
    b=read(H/'RUN_MANIFEST.json')
    if b.get('deadline_epoch') is not None and time.time()>=b['deadline_epoch']:
        raise TimeoutError('Run deadline reached. Completed checkpoints remain; resume never resets the clock.')
    return b
