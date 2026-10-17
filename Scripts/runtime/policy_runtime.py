"""A real pretrained VLA action expert, with differentiable fixed-noise flow mean."""
import os,sys,math,hashlib,contextlib
from pathlib import Path
from control import H,R,read,write,sha,now,asset
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
sys.path.insert(0,str(asset("lerobot")/"src"))
import numpy as np,torch,cv2
from torch.utils.checkpoint import checkpoint
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla,make_att_2d_masks
from data_runtime import TASK

def assets():
    return {"lerobot/smolvla_base":asset("smolvla"),"HuggingFaceTB/SmolVLM2-500M-Video-Instruct":asset("smolvlm")}
def fingerprint(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        h.update(n.encode());h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
class Actor:
    def __init__(self,ckpt=None,resolution=256):
        paths=assets();base=paths["lerobot/smolvla_base"];vlm=paths["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
        cfg=SmolVLAConfig()
        for k,v in read(base/"config.json").items():
            if k not in ("type","input_features","output_features","normalization_mapping") and hasattr(cfg,k):setattr(cfg,k,v)
        cfg.device="cuda";cfg.push_to_hub=False;cfg.load_vlm_weights=False;cfg.vlm_model_name=str(vlm)
        cfg.resize_imgs_with_padding=(resolution,resolution);cfg.num_steps=10;cfg.n_action_steps=50
        cfg.input_features={"observation.state":PolicyFeature(FeatureType.STATE,(7,)),
            "observation.images.front":PolicyFeature(FeatureType.VISUAL,(3,224,224)),
            "observation.images.wrist":PolicyFeature(FeatureType.VISUAL,(3,224,224))}
        cfg.output_features={"action":PolicyFeature(FeatureType.ACTION,(7,))}
        st={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in row.items()} for k,row in read(H/"DATA.json")["features"].items()}
        self.policy=SmolVLAPolicy(cfg,dataset_stats=st)
        load_smolvla(self.policy,base/"model.safetensors",device="cpu",checkpoint_keys_mapping="model._orig_mod.//model.")
        self.model=self.policy.model
        if ckpt:
            c=torch.load(ckpt,map_location="cpu",weights_only=False);missing,extra=self.model.load_state_dict(c["adapter"],strict=False)
            assert not extra and all(k.startswith("vlm_with_expert.vlm.") for k in missing)
        self.policy.cuda().eval()
        self.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
        self.action_mean=torch.tensor(st["action"]["mean"],device="cuda");self.action_std=torch.tensor(st["action"]["std"],device="cuda")
        self.rl_mode=False
    def adapter(self):
        return {n:v.detach().cpu().clone() for n,v in self.model.state_dict().items() if not n.startswith("vlm_with_expert.vlm.")}
    def parameters(self):return [p for p in self.model.parameters() if p.requires_grad]
    def set_rl(self):
        self.rl_mode=True;self.policy.eval();self.model.state_proj.requires_grad_(False)
        self.model.vlm_with_expert.vlm.requires_grad_(False)
    def prep(self,b):
        b=self.policy.normalize_inputs(b)
        images,masks=self.policy.prepare_images(b);state=self.policy.prepare_state(b)
        tokens,lmask=self.policy.prepare_language(b)
        return images,masks,tokens,lmask,state
    def sft_loss(self,b):
        images,masks,tokens,lmask,state=self.prep(b)
        normalized=self.policy.normalize_targets(b);acts=self.policy.prepare_action(normalized)
        losses=self.model(images,masks,tokens,lmask,state,acts)[:,:,:7]
        valid=(~b["action_is_pad"]).float()
        return (losses*valid[:,:,None]).sum()/(valid.sum()*7)
    def mean(self,b,grad=False):
        m=self.model
        # In GRPO the vision/language backbone and state projector are frozen.
        # Cache these only within one observation; action expert remains fully differentiable.
        with torch.no_grad() if self.rl_mode else contextlib.nullcontext():
            images,masks,tokens,lmask,state=self.prep(b)
            pe,pm,pa=m.embed_prefix(images,masks,tokens,lmask,state=state)
            att=make_att_2d_masks(pm,pa);pos=torch.cumsum(pm,dim=1)-1
            _,cache=m.vlm_with_expert.forward(attention_mask=att,position_ids=pos,past_key_values=None,
                inputs_embeds=[pe,None],use_cache=True,fill_kv_cache=True)
        x=torch.zeros((state.shape[0],m.config.chunk_size,m.config.max_action_dim),device=state.device)
        dt=torch.tensor(-1.0/m.config.num_steps,dtype=torch.float32,device=x.device)
        scalar_time=torch.tensor(1.0,dtype=torch.float32,device=x.device)
        for j in range(m.config.num_steps):
            t=scalar_time.expand(len(x))
            if grad:
                v=checkpoint(lambda xx,tt:m.denoise_step(pm,cache,xx,tt),x,t,use_reentrant=False)
            else:v=m.denoise_step(pm,cache,x,t)
            x=x+dt*v
            scalar_time=scalar_time+dt
        return x[:,:,:7].float()
    def native_mean(self,b):
        images,masks,tokens,lmask,state=self.prep(b)
        noise=torch.zeros((len(state),50,32),device=state.device)
        return self.model.sample_actions(images,masks,tokens,lmask,state,noise=noise)[:,:,:7].float()
    def obs_batch(self,obs,instruction=TASK):
        b={"observation.state":torch.as_tensor(obs["observation/state"],dtype=torch.float32,device="cuda")[None],
           "task":[instruction]}
        for src,key in [("observation/image","front"),("observation/wrist_image","wrist")]:
            b["observation.images."+key]=torch.from_numpy(cv2.resize(np.asarray(obs[src]),(224,224),interpolation=cv2.INTER_LINEAR).copy()).permute(2,0,1)[None].float().cuda()/255
        return b
    def physical(self,a):return (a*self.action_std+self.action_mean).detach().cpu().numpy()
    def save(self,path,**kw):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        c=dict(adapter=self.adapter(),data_sha256=sha(H/"DATA.json"),utc=now(),**kw)
        tmp=path.with_suffix(".tmp");torch.save(c,tmp);tmp.replace(path)
        return sha(path)
def log_prob(actions,mean,std):
    return -.5*((actions-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)
def autocast():return torch.autocast("cuda",dtype=torch.bfloat16)
