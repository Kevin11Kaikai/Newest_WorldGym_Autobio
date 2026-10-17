"""Prepare pinned upstreams/assets or check local assets. No implicit network use in training."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from control import R,asset,catalog,config,read,write,sha
from cli import PROFILES,verify_assets,autobio_patch

def matching(path,entry):
    return path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
def upstreams():
    for key,spec in catalog().items():
        if spec['kind']!='git':continue
        p=asset(key)
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--no-checkout',spec['url'],str(p)],check=True)
            subprocess.run(['git','-C',str(p),'checkout','--detach',spec['revision']],check=True)
        current=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if current!=spec['revision']:raise ValueError(f'Existing {p} has another revision; preserving it.')
        if key=='autobio':
            autobio_patch(p,apply=True)
        print(f'{key}: pinned {current}',flush=True)
def download(profile):
    from huggingface_hub import hf_hub_download
    for key in PROFILES[profile]:
        spec=catalog()[key];root=asset(key)
        if spec['kind'] not in ('huggingface','external'):continue
        for rel,entry in spec['files'].items():
            target=root if rel=='.' else root/rel
            if matching(target,entry):continue
            if target.exists():raise ValueError(f'Existing mismatching file preserved: {target}')
            target.parent.mkdir(parents=True,exist_ok=True)
            if spec['kind']=='huggingface':
                # Gated models require the reader's own access; do not accept terms or use mirrors.
                hf_hub_download(spec['repo_id'],filename=rel,revision=spec['revision'],local_dir=str(root))
            else:
                import gdown
                gdown.download(spec['url'],str(target),quiet=False)
            if not matching(target,entry):raise ValueError(f'Download did not match pinned checksum: {target}')
            print(f'{key}/{rel}: SHA256 verified',flush=True)
def decode(download_missing):
    import av,numpy as np,pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    d=config()['data'];episodes=d['splits']['40']+d['validation_episodes'];out=asset('decoded')
    raw=R/'assets/raw';spec=catalog()['decoded'];out.mkdir(parents=True,exist_ok=True)
    for ep in episodes:
        stem=f'episode_{ep:06d}';receipt=out/f'{stem}_verified.json'
        if receipt.exists():
            old=read(receipt)
            for rel,expected in old['decoded_sha256'].items():
                if sha(out/rel)!=expected:raise ValueError(f'Decoded data hash mismatch: {rel}')
            continue
        paths={}
        for role,rel in [('numeric',f'data/chunk-000/{stem}.parquet'),('image',f'videos/chunk-000/image/{stem}.mp4'),('wrist_image',f'videos/chunk-000/wrist_image/{stem}.mp4')]:
            p=raw/rel
            if not p.is_file():
                if not download_missing:raise FileNotFoundError(f'{p}; pass --download to retrieve the pinned dataset.')
                hf_hub_download(spec['dataset'],filename=rel,repo_type='dataset',revision=spec['revision'],local_dir=str(raw))
            paths[role]=p
        table=pq.read_table(paths['numeric']).to_pydict();state=np.stack(table['state']).astype('float32');actions=np.stack(table['actions']).astype('float32')
        ts=np.asarray(table['timestamp'],dtype='float64');n=len(ts)
        assert state.shape==actions.shape==(n,7) and np.isfinite(state).all() and np.isfinite(actions).all()
        assert np.all(np.array(table['episode_index'])==ep) and np.array_equal(table['frame_index'],np.arange(n)) and max(abs(ts-np.arange(n)/50))<1e-4
        target=out/f'{stem}_numeric.npz'
        if target.exists():raise FileExistsError(f'Incomplete decode output preserved: {target}; inspect it before starting in a new directory.')
        np.savez(target,state=state,actions=actions,timestamps=ts)
        emitted=[target]
        for camera in ['image','wrist_image']:
            dst=out/f'{stem}_{camera}.npy';arr=np.lib.format.open_memmap(dst,mode='w+',dtype='uint8',shape=(n,224,224,3))
            count=0
            with av.open(str(paths[camera])) as container:
                stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2
                assert abs(float(stream.average_rate)-50)<1e-3
                for i,frame in enumerate(container.decode(stream)):
                    if i>=n:raise ValueError('Video longer than numeric sequence')
                    img=frame.to_ndarray(format='rgb24');assert img.shape==(224,224,3)
                    assert frame.time is not None and abs(float(frame.time)-ts[i])<.011
                    arr[i]=img;count+=1
            assert count==n;arr.flush();del arr;emitted.append(dst)
        write(receipt,dict(dataset=spec['dataset'],revision=spec['revision'],episode=ep,frames=n,
            source_sha256={k:sha(v) for k,v in paths.items()},decoded_sha256={p.name:sha(p) for p in emitted}))
        print(f'{stem}: {n} paired frames decoded',flush=True)
def bundle(location,export):
    location=location.resolve()
    for key,spec in catalog().items():
        if spec['kind']!='task':continue
        for rel,entry in spec['files'].items():
            target_rel=Path(spec['path']).relative_to('assets/task')
            if rel!='.':target_rel=target_rel/rel
            local=asset(key) if rel=='.' else asset(key)/rel
            src,dst=(local,location/target_rel) if export else (location/target_rel,local)
            if not matching(src,entry):raise ValueError(f'Task asset missing or checksum mismatch: {src}')
            if dst.exists():
                if not matching(dst,entry):raise FileExistsError(f'Preserving mismatching destination: {dst}')
            else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    print('Verified task bundle copied. This does not publish or upload it.')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,help='Optional asset path overrides; keep local.')
    sp=ap.add_subparsers(dest='command',required=True)
    sp.add_parser('upstreams',help='Clone exact upstream commits and apply the two-line AutoBio compatibility patch.')
    c=sp.add_parser('check');c.add_argument('--profile',choices=PROFILES,default='sft');c.add_argument('--full-hash',action='store_true')
    p=sp.add_parser('public-models');p.add_argument('--profile',choices=PROFILES,default='sft')
    p=sp.add_parser('data');p.add_argument('--download',action='store_true')
    for cmd in ['import-task','export-task']:
        p=sp.add_parser(cmd);p.add_argument('directory',type=Path)
    args=ap.parse_args()
    if args.assets:os.environ['WORLDGYM_ASSETS']=str(args.assets.resolve())
    if args.command=='upstreams':upstreams()
    elif args.command=='check':print(json.dumps(verify_assets(args.profile,args.full_hash),indent=2))
    elif args.command=='public-models':download(args.profile)
    elif args.command=='data':decode(args.download)
    else:bundle(args.directory,args.command=='export-task')
if __name__=='__main__':main()
