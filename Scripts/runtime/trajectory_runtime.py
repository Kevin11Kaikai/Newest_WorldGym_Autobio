"""Auditable 22.6s trajectories. Imagined loop receives reset observation only."""
import time,json
from pathlib import Path
from control import H,R,read,write,sha,budget,now
from policy_runtime import autocast,log_prob
from data_runtime import TASK
import numpy as np,torch,cv2
def save_initial(seed,physical):
    p=H/"initial_observations"/f"{seed}.npz";p.parent.mkdir(exist_ok=True)
    if not p.exists():
        o=physical.reset(seed);np.savez(p,front=o["observation/image"],wrist=o["observation/wrist_image"],
            state=o["observation/state"],bounds=physical.bounds)
    z=np.load(p)
    return {"observation/image":z["front"].copy(),"observation/wrist_image":z["wrist"].copy(),"observation/state":z["state"].copy()},z["bounds"].copy()
def rollout(actor,env,seed,folder,native=False,noise_seed=None,std=.05,instruction=TASK,initial=None,bounds=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);marker=folder/"trajectory.json"
    if marker.exists():
        old=read(marker);assert old["seed"]==seed and old["instruction"]==instruction
        assert old["policy_hash"]==actor.run_fingerprint
        assert old["native"]==native and old["noise_seed"]==noise_seed and old["std"]==std
        assert sha(folder/"policy_trace.npz")==old["trace_sha256"]
        return old
    t=time.time()
    if native:
        assert initial is not None and bounds is not None
        obs=env.reset(initial,seed+711)
    else:obs=env.reset(seed);bounds=env.bounds
    fronts=[obs["observation/image"].copy()];wrists=[obs["observation/wrist_image"].copy()];states=[obs["observation/state"].copy()]
    # Camera resolution is allowed to differ at reset; all saved images use224x224.
    resize=lambda x:cv2.resize(x,(224,224),interpolation=cv2.INTER_LINEAR)
    fronts=[resize(fronts[0])];wrists=[resize(wrists[0])]
    trace={k:[] for k in ["front","wrist","state","sample","mean","old_logprob","valid","controls"]}
    rng=torch.Generator().manual_seed(int(noise_seed)) if noise_seed is not None else None
    tick=0
    while tick<113:
        budget(1200)
        with torch.inference_mode(),autocast():mu=actor.mean(actor.obs_batch(obs,instruction))[0].float().cpu()
        sample=mu+std*torch.randn(mu.shape,generator=rng) if rng is not None else mu.clone()
        valid=np.arange(50)<min(50,(113-tick)*10)
        trace["front"].append(resize(obs["observation/image"]));trace["wrist"].append(resize(obs["observation/wrist_image"]))
        trace["state"].append(obs["observation/state"].copy());trace["sample"].append(sample.numpy())
        trace["mean"].append(mu.numpy());trace["old_logprob"].append(log_prob(sample,mu,std).numpy());trace["valid"].append(valid)
        control=actor.physical(sample.cuda())[valid]
        control=np.clip(control,bounds[:,0],bounds[:,1]);trace["controls"].extend(control.copy())
        obs=env.step(control);tick+=len(control)//10
        if native:
            for x,q in zip(env.last_images,env.last_states):
                fronts.append(resize(x[:,:256]));wrists.append(resize(x[:,256:]));states.append(q.copy())
        else:
            for x in env.last_observations:
                fronts.append(resize(x["observation/image"]));wrists.append(resize(x["observation/wrist_image"]));states.append(x["observation/state"].copy())
    assert len(fronts)==114 and len(trace["controls"])==1130
    np.savez_compressed(folder/"policy_trace.npz",**{k:np.asarray(v) for k,v in trace.items()})
    np.save(folder/"front_rgb.npy",np.asarray(fronts));np.save(folder/"wrist_rgb.npy",np.asarray(wrists));np.save(folder/"proprio.npy",np.asarray(states))
    result=dict(policy_hash=actor.run_fingerprint,seed=int(seed),instruction=instruction,native=native,noise_seed=noise_seed,std=std,seconds=time.time()-t,
        frames=len(fronts),controls=1130,policy_calls=len(trace["mean"]),trace_sha256=sha(folder/"policy_trace.npz"),
        generated_future_input=native,future_true_observation_in_native=False,utc=now())
    if not native:
        result.update(success=bool(env.success),first_success=env.first_success)
        import mujoco
        audit=dict(heights=env.audit_height,contacts=env.audit_contacts,official_hits=env.audit_hits,
            finger_geom=int(mujoco.mj_name2id(env.task.model,mujoco.mjtObj.mjOBJ_GEOM,"1/aloha:leftfinger")),
            cap_geom=int(mujoco.mj_name2id(env.task.model,mujoco.mjtObj.mjOBJ_GEOM,"1/cap_cyl")))
        write(folder/"official_audit.json",audit)
    write(marker,result);return result
def trace_batch(trace,i):
    return {"observation.state":torch.from_numpy(trace["state"][i].copy()).float()[None].cuda(),"task":[TASK],
        "observation.images.front":torch.from_numpy(trace["front"][i].copy()).permute(2,0,1)[None].float().cuda()/255,
        "observation.images.wrist":torch.from_numpy(trace["wrist"][i].copy()).permute(2,0,1)[None].float().cuda()/255}
