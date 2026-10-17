"""Train-only AutoBio normalization and deterministic episode sampling."""
import sys,collections
from pathlib import Path
from control import H,R,read,write,sha,now,asset,config

import numpy as np,torch
C=asset("decoded")
TASK="Pick up the orange-capped test tube from the rack."
class Data:
    def __init__(self,condition=40):
        p=config()["data"]
        self.fit=p["splits"][str(condition)];self.val=p["validation_episodes"]
        self.numeric={};self.images=collections.OrderedDict()
        for ep in self.fit+self.val:
            z=np.load(C/f"episode_{ep:06d}_numeric.npz");self.numeric[ep]={"state":z["state"],"actions":z["actions"]}
        self.events={}
        for ep,z in self.numeric.items():
            a=z["actions"];closed=np.isclose(a[:,6],.013,atol=1e-5)
            on=np.flatnonzero(closed & np.r_[True,~closed[:-1]])
            self.events[ep]=np.unique(np.concatenate([np.arange(max(0,i-50),min(len(a)-1,i+11)) for i in on])) if len(on) else np.arange(len(a)-1)
        features={}
        for name,key in [("observation.state","state"),("action","actions")]:
            x=np.concatenate([self.numeric[e][key] for e in self.fit]).astype("float64")
            features[name]={k:v.astype("float32").tolist() for k,v in dict(mean=x.mean(0),std=np.maximum(x.std(0),1e-5)).items()}
        self.stats=features
        self.receipt=dict(status="FIT_ONLY_STATS",utc=now(),condition=condition,fit=self.fit,validation=self.val,
            excluded_heldout=p["heldout_demos_never_used"],features=features,
            numeric_sha256={str(ep):sha(C/f"episode_{ep:06d}_numeric.npz") for ep in self.fit+self.val},
            inherited_decoded_provenance="Pinned AutoBio dataset; decoded source receipts and numeric hashes",
            action_alignment="actions[frame+1:frame+51], as previously verified physical execution")
        if (H/"DATA.json").exists():
            old=read(H/"DATA.json")
            for k in ("condition","fit","validation","features","numeric_sha256"):assert old[k]==self.receipt[k],k
        else:write(H/"DATA.json",self.receipt)
    def sample(self,ep,t):
        z=self.numeric[ep];n=len(z["state"]);idx=t+1+np.arange(50);pad=idx>=n;idx=np.minimum(idx,n-1)
        b={"observation.state":torch.from_numpy(z["state"][t].copy()),
           "action":torch.from_numpy(z["actions"][idx].copy()),"action_is_pad":torch.from_numpy(pad)}
        for cam,key in [("image","front"),("wrist_image","wrist")]:
            pair=(ep,cam)
            if pair not in self.images:self.images[pair]=np.load(C/f"episode_{ep:06d}_{cam}.npy",mmap_mode="r")
            self.images.move_to_end(pair)
            while len(self.images)>20:self.images.popitem(last=False)
            b["observation.images."+key]=torch.from_numpy(self.images[pair][t].copy()).permute(2,0,1).float()/255
        return b
    def batch(self,step,seed,size=4,validation=False):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(step)]));rows=[]
        for _ in range(size):
            ep=int(rng.choice(self.val if validation else self.fit));n=len(self.numeric[ep]["state"])
            t=int(rng.choice(self.events[ep])) if not validation and rng.random()<.5 else int(rng.integers(n-1))
            rows.append(self.sample(ep,t))
        b={k:torch.stack([x[k] for x in rows]).cuda() for k in rows[0]}
        b["task"]=[TASK]*size;return b
