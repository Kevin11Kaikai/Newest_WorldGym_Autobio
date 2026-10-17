"""Actual GRPO on a pretrained language-conditioned VLA, matched across simulated environments."""
import os,time,json,gc,importlib
from pathlib import Path
from control import H,R,read,write,sha,append,state,budget,now,log
from rl_precision import RLActor as Actor
from policy_runtime import autocast,log_prob,fingerprint
from environment_runtime import Physical,Native
from trajectory_runtime import save_initial,rollout,trace_batch
import torch,numpy as np
def score_trajectory(reward,folder,p):
    mark=folder/"reward.json"
    if mark.exists():return read(mark)
    f=np.load(folder/"front_rgb.npy");w=np.load(folder/"wrist_rgb.npy")
    instruction=read(folder/"trajectory.json")["instruction"]
    scores=[dict(tick=t,**reward.score(f[t],w[t],instruction=instruction,initial=f[0])) for t in p["reward"]["snapshot_indices"]]
    r=dict(score=max(x["score"] for x in scores),frames=scores,source_trace_sha256=sha(folder/"policy_trace.npz"),
        official_success_not_read=True,utc=now())
    write(mark,r);return r
def optimizer_sample(actor,opt):
    records=[]
    for name,param in actor.model.named_parameters():
        if not param.requires_grad or param.grad is None or len(records)>=12:continue
        assert param.dtype==torch.float32,"GRPO master parameters must be FP32"
        if not torch.isfinite(param.grad).all():raise ValueError("Nonfinite gradient "+name)
        flat=param.detach().flatten();grad=param.grad.detach().flatten();ids=torch.linspace(0,len(flat)-1,min(32,len(flat))).long().to(flat.device)
        st=opt.state.get(param,{})
        records.append(dict(name=name,indices=ids.cpu().numpy(),before=flat[ids].cpu().numpy(),gradient=grad[ids].cpu().numpy(),
            m=(st["exp_avg"].flatten()[ids].cpu().numpy() if "exp_avg" in st else np.zeros(len(ids),np.float32)),
            v=(st["exp_avg_sq"].flatten()[ids].cpu().numpy() if "exp_avg_sq" in st else np.zeros(len(ids),np.float32)),
            step=int(st.get("step",0))+1))
    return records
def update(actor,reference,opt,folders,rewards,cfg,group,out):
    values=np.asarray(rewards,np.float64);sd=float(values.std(ddof=1))
    advantages=(values-values.mean())/(sd+1e-6);before=fingerprint(actor.adapter())
    trajectories=[dict(np.load(f/"policy_trace.npz")) for f in folders]
    instructions=[read(f/"trajectory.json")["instruction"] for f in folders]
    chosen=[]
    for k,z in enumerate(trajectories):
        rng=np.random.default_rng(np.random.SeedSequence([cfg["training_seed"],group,k]))
        chosen.append(sorted(rng.choice(len(z["sample"]),min(cfg["observations_per_trajectory"],len(z["sample"])),replace=False).tolist()))
    if sd<1e-12:
        return dict(status="ZERO_ADVANTAGE_RETAINED",rewards=list(map(float,rewards)),advantages=advantages.tolist(),
            weights_before=before,weights_after=before,actual_update=False,selected_indices=chosen)
    updates=[]
    for epoch in range(cfg["epochs_per_group"]):
        budget(1800);opt.zero_grad(set_to_none=True);loss_sum=0.;pg_sum=0.;kl_sum=0.;clip_sum=0.;lp_error=0.
        for k,z in enumerate(trajectories):
            for j in chosen[k]:
                b=trace_batch(z,j);b["task"]=[instructions[k]]
                acts=torch.from_numpy(z["sample"][j]).cuda();old=torch.from_numpy(z["old_logprob"][j]).cuda()
                mask=torch.from_numpy(z["valid"][j]).cuda()[:,None].float()
                with torch.no_grad(),autocast():ref=reference.mean(b)[0]
                with autocast():mu=actor.mean(b,grad=True)[0]
                lp=log_prob(acts,mu,cfg["std"])
                if epoch==0:
                    err=float((lp.detach()-old).abs().max());lp_error=max(lp_error,err)
                    assert err<.001,("rollout likelihood mismatch",err)
                ratio=(lp-old).clamp(-20,20).exp();adv=float(advantages[k])
                pg=-torch.minimum(ratio*adv,ratio.clamp(1-cfg["clip_low"],1+cfg["clip_high"])*adv)
                kl=.5*((mu-ref)/cfg["std"]).square()
                denominator=mask.sum()*7
                pg_mean=(pg*mask).sum()/denominator;kl_mean=(kl*mask).sum()/denominator
                loss=(pg_mean+cfg["kl_beta"]*kl_mean)/(len(folders)*len(chosen[k]))
                if not torch.isfinite(loss):raise ValueError("Nonfinite GRPO loss")
                loss.backward();loss_sum+=float(loss);pg_sum+=float(pg_mean)/len(folders)/len(chosen[k])
                kl_sum+=float(kl_mean)/len(folders)/len(chosen[k])
                clip_sum+=float((((ratio<1-cfg["clip_low"])|(ratio>1+cfg["clip_high"])).float()*mask).sum()/denominator)/len(folders)/len(chosen[k])
                if k==0 and j==chosen[0][0]:
                    np.savez(out/f"likelihood_epoch{epoch}.npz",actions=z["sample"][j],old_mean=z["mean"][j],
                        old_lp=z["old_logprob"][j],new_mean=mu.detach().float().cpu().numpy(),
                        ref_mean=ref.float().cpu().numpy(),valid=z["valid"][j],advantage=adv,
                        pg=float(pg_mean),kl=float(kl_mean),std=cfg["std"],clip_low=cfg["clip_low"],clip_high=cfg["clip_high"])
        grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg["gradient_clip"])
        assert torch.isfinite(grad);sample=optimizer_sample(actor,opt);opt.step()
        params=dict(actor.model.named_parameters())
        arrays={}
        for i,row in enumerate(sample):
            row["after"]=params[row["name"]].detach().flatten()[torch.from_numpy(row["indices"]).cuda()].cpu().numpy()
            for key in ["indices","before","gradient","m","v","after"]:arrays[f"{i}_{key}"]=row[key]
        np.savez(out/f"adam_epoch{epoch}.npz",**arrays)
        write(out/f"adam_epoch{epoch}.json",dict(parameters=[dict(name=r["name"],step=r["step"]) for r in sample],
            lr=opt.param_groups[0]["lr"],betas=opt.param_groups[0]["betas"],eps=opt.param_groups[0]["eps"],weight_decay=0))
        updates.append(dict(epoch=epoch,loss=loss_sum,policy_loss=pg_sum,kl=kl_sum,clipped_fraction=clip_sum,
            gradient_norm=float(grad),rollout_logprob_max_error=lp_error))
    after=fingerprint(actor.adapter());assert after!=before
    return dict(status="ACTUAL_VLA_GRPO_UPDATE",rewards=list(map(float,rewards)),advantages=advantages.tolist(),
        weights_before=before,weights_after=after,actual_update=True,selected_indices=chosen,epochs=updates)
def train(stage,base_checkpoint,native,groups,world_checkpoint=None,seeds=None,instruction=None):
    p=read(H/"protocol.json");cfg=p["rl"];budget(1800);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    folder=H/"rl"/stage;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"COMPLETE.json").exists():return read(folder/"COMPLETE.json")
    last=folder/"last.pt";step0=0
    a=Actor(last if last.exists() else base_checkpoint);a.set_rl()
    ref=Actor(p["reference_checkpoint"]);ref.set_rl();ref.model.requires_grad_(False)
    opt=torch.optim.AdamW(a.parameters(),lr=cfg["lr"],weight_decay=0.)
    if last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False);assert ck["protocol_sha256"]==sha(H/"protocol.json")
        opt.load_state_dict(ck["optimizer"]);step0=ck["group"]
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
    module=importlib.import_module(p["reward"]["module"]);reward=getattr(module,p["reward"]["class_name"])(device="cpu")
    physical=Physical()
    seeds=seeds or p["scene_seeds"]["rl_fit"];instruction=instruction or p["instruction"]
    for seed in seeds:save_initial(seed,physical)
    if native:physical.close();env=Native(world_checkpoint)
    else:env=physical
    start=time.time();completed=[]
    for group in range(step0,groups):
        budget(3000);s=time.time();seed=seeds[group%len(seeds)];out=folder/f"group_{group+1:03d}";out.mkdir(exist_ok=True)
        state(stage,group=group+1,total=groups,phase="rollout")
        a.policy.cuda();ref.policy.cuda()
        if native:env.to("cuda")
        a.run_fingerprint=fingerprint(a.adapter());fs=[];initial,bounds=save_initial(seed,None)
        for k in range(cfg["group_size"]):
            path=out/f"rollout_{k}"
            rollout(a,env,seed,path,native=native,noise_seed=cfg["training_seed"]+100*group+k,std=cfg["std"],
                instruction=instruction,initial=initial,bounds=bounds)
            fs.append(path)
        # Reward model and actor/world model are phased to respect16GB memory.
        if native:env.to("cpu")
        a.policy.cpu();ref.policy.cpu();gc.collect();torch.cuda.empty_cache()
        reward.to("cuda");state(stage,group=group+1,total=groups,phase="VLM_reward")
        rewards=[score_trajectory(reward,f,p)["score"] for f in fs]
        reward.to("cpu");gc.collect();torch.cuda.empty_cache();a.policy.cuda();ref.policy.cuda()
        state(stage,group=group+1,total=groups,phase="GRPO_update")
        result=update(a,ref,opt,fs,rewards,cfg,group,out);result.update(group=group+1,seed=seed,seconds=time.time()-s,utc=now())
        write(out/"update.json",result);completed.append(result)
        a.save(last,optimizer=opt.state_dict(),group=group+1,protocol_sha256=sha(H/"protocol.json"),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),result=result)
        if group==0 or group+1 in [12,24,groups]:
            a.save(folder/f"group_{group+1:03d}.pt",group=group+1,protocol_sha256=sha(H/"protocol.json"),result=result)
        print(json.dumps(dict(stage=stage,group=group+1,total=groups,rewards=rewards,update=result["actual_update"],seconds=result["seconds"])),flush=True)
    endpoint=folder/"endpoint.pt";a.save(endpoint,group=groups,protocol_sha256=sha(H/"protocol.json"))
    allrows=[read(folder/f"group_{i:03d}/update.json") for i in range(1,groups+1)]
    done=dict(status="COMPLETE",stage=stage,groups=groups,actual_update_groups=sum(x["actual_update"] for x in allrows),
        endpoint=str(endpoint.relative_to(H)),sha256=sha(endpoint),seconds=sum(x["seconds"] for x in allrows),
        reward_is_official_success=False,utc=now())
    write(folder/"COMPLETE.json",done);env.close()
    del a,ref,opt,reward;gc.collect();torch.cuda.empty_cache()
    return done
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("stage");ap.add_argument("base");ap.add_argument("--native",action="store_true");ap.add_argument("--groups",type=int,default=24)
    args=ap.parse_args();print(json.dumps(train(args.stage,H/args.base,args.native,args.groups)))
