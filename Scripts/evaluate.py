"""Evaluate a complete fixed scene matrix, or select SFT checkpoints using development scenes only."""
import argparse,gc,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from cli import common,begin,configure,resolve_checkpoint,gpu_lock,finish
from control import read,write,sha,config

def main():
    ap=argparse.ArgumentParser(description=__doc__);common(ap)
    ap.add_argument('--checkpoint',action='append',help='Repeat for a paired matrix; saved statistics are written only after all rows complete.')
    ap.add_argument('--stats',help='Matching DATA.json, or asset name normalization.')
    ap.add_argument('--policy',choices=['smolvla','act'],default='smolvla',help='ACT accepts a full pretrained directory; SmolVLA accepts an adapter .pt.')
    ap.add_argument('--environment',choices=['mujoco','worldgym'],default='mujoco')
    ap.add_argument('--split',choices=['selection_dev','final'],default='selection_dev')
    ap.add_argument('--sft-run',type=Path,help='Select among this SFT run\'s checkpoints on all development seeds, then write sft/selected.pt.')
    ap.add_argument('--vlm-reward',action='store_true',help='Also score saved RGB with the task-domain VLM. This is not official success.')
    ap.add_argument('--smoke',action='store_true',help='One fixed development scene; no effectiveness claim.')
    args=ap.parse_args();configure(args)
    if args.sft_run:
        if args.checkpoint or args.environment!='mujoco' or args.split!='selection_dev' or args.smoke or args.policy!='smolvla':
            ap.error('--sft-run requires the complete MuJoCo development matrix, without --checkpoint/--smoke.')
        rows=[json.loads(x) for x in (args.sft_run/'sft/checkpoints.jsonl').read_text().splitlines()]
        checkpoints=[(args.sft_run/x['path']).resolve() for x in rows]
        for cp,row in zip(checkpoints,rows):
            if sha(cp)!=row['sha256']:raise ValueError('SFT checkpoint hash changed')
        stats=args.sft_run/'DATA.json'
    else:
        if not args.checkpoint or args.policy=='smolvla' and not args.stats:ap.error('--checkpoint and (for SmolVLA) --stats are required.')
        checkpoints=[resolve_checkpoint(x) for x in args.checkpoint];stats=args.stats;rows=[]
    names=[f'policy_{i:02d}' for i in range(len(checkpoints))]
    cfg=config();seeds=cfg['scene_seeds']['selection_dev' if args.smoke else args.split]
    if args.smoke:seeds=seeds[:1]
    settings=dict(command='evaluate',policy=args.policy,environment=args.environment,split=args.split,smoke=args.smoke,vlm_reward=args.vlm_reward,
                  checkpoints=[dict(name=n,sha256=sha(cp/'model.safetensors' if args.policy=='act' else cp),
                                    config_sha256=sha(cp/'config.json') if args.policy=='act' else None)
                               for n,cp in zip(names,checkpoints)],seeds=seeds,select_sft=bool(args.sft_run))
    profile=('rl' if args.environment=='worldgym' else 'mujoco_rl') if args.vlm_reward else ('native' if args.environment=='worldgym' else 'mujoco')
    if args.policy=='act' and not args.vlm_reward:profile='act_native' if args.environment=='worldgym' else 'act_mujoco'
    with gpu_lock():
        begin(args,profile,settings,stats)
        import numpy as np,torch
        from rl_precision import RLActor
        from policy_runtime import fingerprint
        from environment_runtime import Physical,Native
        from trajectory_runtime import rollout,save_initial
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
        allrows=[]
        for name,cp in zip(names,checkpoints):
            if args.policy=='act':
                from act_policy import ACTActor
                actor=ACTActor(cp)
            else:actor=RLActor(cp);actor.run_fingerprint=fingerprint(actor.adapter())
            actor.policy.reset()
            physical=Physical()
            if args.environment=='worldgym':
                for seed in seeds:save_initial(seed,physical)
                physical.close();env=Native()
            else:env=physical
            paths=[]
            for seed in seeds:
                path=args.run/'trajectories'/name/str(seed)
                initial,bounds=save_initial(seed,None) if args.environment=='worldgym' else (None,None)
                result=rollout(actor,env,seed,path,native=args.environment=='worldgym',initial=initial,bounds=bounds)
                allrows.append(dict(policy=name,**result));paths.append(path)
                print(json.dumps(dict(coverage=len(allrows),expected=len(checkpoints)*len(seeds))),flush=True)
            env.close();actor.policy.cpu();del actor,env;gc.collect();torch.cuda.empty_cache()
            if args.vlm_reward:
                from reward_adapter_runtime import AdapterReward
                from train_grpo import score_trajectory
                reward=AdapterReward(device='cuda')
                for row,path in zip(allrows[-len(seeds):],paths):row['vlm_reward']=score_trajectory(reward,path,cfg)['score']
                del reward;gc.collect();torch.cuda.empty_cache()
        assert len(allrows)==len(checkpoints)*len(seeds)
        summary={'status':'COMPLETE','environment':args.environment,'scenes':len(seeds),'policies':settings['checkpoints'],
                 'scope':'Historical-scene reproduction or engineering smoke check; not a fresh independent confirmation.',
                 'rows':allrows,'official_success_available':args.environment=='mujoco'}
        if args.environment=='mujoco':
            summary['successes']={n:sum(r['success'] for r in allrows if r['policy']==n) for n in names}
            base=np.array([r['success'] for r in allrows if r['policy']==names[0]],int)
            summary['paired_vs_first']={}
            for n in names[1:]:
                a=np.array([r['success'] for r in allrows if r['policy']==n],int);d=a-base
                rng=np.random.default_rng(9011);boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(20000)])
                summary['paired_vs_first'][n]=dict(delta_pp=float(d.mean()*100),wins=int((d==1).sum()),harms=int((d==-1).sum()),
                     scene_bootstrap95_pp=(100*np.quantile(boots,[.025,.975])).tolist())
        if rows:
            chosen=min(range(len(rows)),key=lambda i:(-summary['successes'][names[i]],rows[i]['validation'],rows[i]['step']))
            target=args.sft_run/'sft/selected.pt'
            if target.exists() and sha(target)!=sha(checkpoints[chosen]):raise FileExistsError('Existing selected checkpoint differs; never silently replace it.')
            if not target.exists():shutil.copyfile(checkpoints[chosen],target)
            receipt=dict(status='DEVELOPMENT_SELECTED',chosen_step=rows[chosen]['step'],checkpoint_sha256=sha(target),
                         rule=cfg['sft']['selection'],development_matrix=summary['successes'],final_labels_used=False)
            write(args.sft_run/'sft/SELECTION.json',receipt);summary['selection']=receipt
        finish(args.run,summary)
if __name__=='__main__':main()
