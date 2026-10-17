"""Matched ACT fine-tuning; preserves the original 10-demo normalization."""
import os,sys,time,json,math,random,csv,hashlib
from pathlib import Path
from control import H as HERE,R as ROOT,asset,read as read_json,write as write_json,sha as sha256,budget
sys.path.insert(0,str(asset('lerobot')/'src'))
import numpy as np,torch
from torch.utils.data import DataLoader,Sampler
from act_data import EpisodeDataset,event_indices,validation_anchors,to_device,augment
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
def load_act(path):
    cfg=PreTrainedConfig.from_pretrained(path,local_files_only=True)
    cfg.pretrained_backbone_weights=None
    return ACTPolicy.from_pretrained(path,config=cfg,local_files_only=True).cuda()
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(s):
    random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"]);torch.cuda.set_rng_state_all(s["cuda"])
class Batches(Sampler):
    def __init__(self,ds,start,end,cfg,seed):
        self.n=len(ds);self.start=start;self.end=end;self.cfg=cfg;self.seed=seed;self.events=event_indices(ds)
    def __len__(self):return self.end-self.start
    def __iter__(self):
        for step in range(self.start,self.end):
            r=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
            ix=r.integers(self.n,size=self.cfg["batch_size"])
            m=r.random(len(ix))<self.cfg["event_probability"]
            ix[m]=r.choice(self.events,size=int(m.sum()))
            yield ix.tolist()
@torch.inference_mode()
def validate(policy,loader,std):
    policy.eval();n=0.;d=0
    for b in loader:
        b=to_device(b);a=policy.predict_action_chunk(b);mask=(~b["action_is_pad"]).unsqueeze(-1)
        n+=float((((a-b["action"])/std).abs()*mask).sum());d+=int(mask.sum())*7
    return n/d
def check_deadline():budget()
def save(folder,policy,opt,step,p,seed,condition,validation):
    folder.mkdir(parents=True,exist_ok=True)
    ck=dict(model=policy.state_dict(),optimizer=opt.state_dict(),step=step,rng=rng_state(),
        condition=condition,training_seed=seed,protocol_sha256=sha256(HERE/"protocol.json"),
        source_sha256=p["base_act"]["checkpoint_sha256"],validation_prior_l1=validation)
    tmp=folder/"training_state.tmp";torch.save(ck,tmp);tmp.replace(folder/"training_state.pt")
    policy.save_pretrained(folder/"pretrained")
    write_json(folder/"receipt.json",dict(step=step,condition=condition,training_seed=seed,
        model_sha256=sha256(folder/"pretrained/model.safetensors"),
        training_state_sha256=sha256(folder/"training_state.pt"),protocol_sha256=ck["protocol_sha256"]))
def run(condition,seed,p,pilot=False):
    cfg=p["sft"];seed_all(seed)
    ds=EpisodeDataset(cfg["splits"][str(condition)])
    for ep in ds.episode_ids:
        probe=EpisodeDataset([ep]);b=probe[len(probe)-1]
        assert b["action_is_pad"].sum()==49 and torch.equal(b["action"][0],b["action"][-1])
    source=ROOT/p["base_act"]["directory"]
    assert sha256(source/"model.safetensors")==p["base_act"]["checkpoint_sha256"]
    policy=load_act(source)
    opt=torch.optim.AdamW(policy.get_optim_params(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    for group in opt.param_groups:group["lr"]=cfg["learning_rate"]
    out=HERE/"sft_act"
    out.mkdir(parents=True,exist_ok=True);latest=out/"last/training_state.pt"
    start=0;end=2 if pilot else cfg["steps"]
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        assert ck["protocol_sha256"]==sha256(HERE/"protocol.json")
        assert ck["source_sha256"]==p["base_act"]["checkpoint_sha256"]
        policy.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);start=ck["step"];restore(ck["rng"])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!="step":state[k]=v.cuda()
        del ck
    loader=DataLoader(ds,batch_sampler=Batches(ds,start,end,cfg,seed),num_workers=2,
        pin_memory=True,persistent_workers=True,prefetch_factor=2,generator=torch.Generator().manual_seed(15001))
    valid=EpisodeDataset(cfg["validation_episodes"],anchors=validation_anchors(cfg["validation_episodes"]))
    vl=DataLoader(valid,batch_size=cfg["batch_size"],num_workers=2,pin_memory=True,
        persistent_workers=True,generator=torch.Generator().manual_seed(15002))
    std=torch.tensor(read_json(asset("act_stats"))["features"]["action"]["std"],device="cuda")
    log=out/"updates.jsonl";begin=time.time();window=time.time();previous=start;val=None;validation_step=None;step=start
    before=next(v for v in policy.parameters() if v.requires_grad).detach().cpu().clone() if pilot else None
    torch.cuda.reset_peak_memory_stats()
    with log.open("a",encoding="utf-8") as f:
        for step,b in enumerate(loader,start+1):
            policy.train();b=augment(to_device(b),cfg["augmentation"])
            loss,parts=policy(b)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg["grad_clip"])
            assert torch.isfinite(gn);opt.step()
            if step%100==0 or step==end:
                record=dict(step=step,loss=float(loss),grad_norm=float(gn),parts=parts,
                    seconds_per_update=(time.time()-window)/(step-previous),
                    elapsed_seconds=time.time()-begin,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
                f.write(json.dumps(record)+"\n");f.flush();print(json.dumps(dict(event="SFT_UPDATE",condition=condition,seed=seed,**record)),flush=True)
                write_json(HERE/"RUN_STATE.json",dict(status="RUNNING",stage="pilot_sft" if pilot else "sft",
                    condition=condition,seed=seed,step=step,total=end,final_labels_accessed=False))
                window=time.time();previous=step
            if not pilot and (step%cfg["validation_every"]==0 or step==end):
                rs=rng_state();val=validate(policy,vl,std);validation_step=step;restore(rs)
                with (out/"validation.jsonl").open("a") as vf:vf.write(json.dumps(dict(step=step,prior_l1=val))+"\n")
            if not pilot and (step%cfg["checkpoint_every"]==0 or step==end or step in cfg["candidate_steps"]):
                save(out/"last",policy,opt,step,p,seed,condition,val)
                check_deadline()
            if not pilot and step in cfg["candidate_steps"]:
                dest=out/f"step_{step:06d}";dest.mkdir(exist_ok=True)
                policy.save_pretrained(dest/"pretrained")
                write_json(dest/"receipt.json",dict(step=step,condition=condition,training_seed=seed,
                    model_sha256=sha256(dest/"pretrained/model.safetensors"),validation_prior_l1=val,validation_metric_step=validation_step))
    if pilot:
        assert not torch.equal(before,next(v for v in policy.parameters() if v.requires_grad).detach().cpu())
        dest=out/"reload";policy.eval();policy.save_pretrained(dest)
    else:
        dest=out/"last/pretrained";policy.eval()
    reloaded=load_act(dest).eval()
    batch=to_device(next(iter(vl)))
    with torch.inference_mode():
        a=policy.predict_action_chunk(batch);b=reloaded.predict_action_chunk(batch)
    assert torch.equal(a,b),"Checkpoint reload changed action outputs"
    receipt=dict(status="COMPLETE",condition=condition,training_seed=seed,start_step=start,end_step=step,
        seconds=time.time()-begin,exact_reload=True,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
        train_episode_ids=ds.episode_ids,validation_episode_ids=cfg["validation_episodes"],
        final_labels_accessed=False,source_model_sha256=p["base_act"]["checkpoint_sha256"])
    write_json(out/"COMPLETE.json",receipt);print(json.dumps(receipt),flush=True)
    return receipt
