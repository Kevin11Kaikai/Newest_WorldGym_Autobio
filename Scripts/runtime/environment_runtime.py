"""Physical ground truth and separate native video model. No future physics in imagined rollouts."""
import os,sys,time,gc
os.environ.setdefault("MUJOCO_GL","egl")
from control import H,R,read,write,sha,budget,asset,config
sys.path.insert(0,str(asset("autobio")/"autobio"))
import numpy as np,torch,cv2
from task import create_task
from evaluator import Evaluator

def panorama(front,wrist):
    x=np.concatenate([cv2.resize(front,(256,256)),cv2.resize(wrist,(256,256))],1)
    x[:32]=0;return x
def copyobs(o):
    return {k:np.asarray(o[k]).copy() for k in ["observation/image","observation/wrist_image","observation/state"]}
class Physical:
    def __init__(self):
        self.task=create_task("pickup");self.ev=Evaluator(self.task)
        self.bounds=self.task.model.actuator_ctrlrange[:7].copy()
    def reset(self,seed):
        self.task.reset(int(seed));self.ev.reset();self.steps=0
        self.success=bool(self.task.check());self.first_success=0. if self.success else None
        self.audit_height=[];self.audit_contacts=[];self.audit_hits=[]
        return copyobs(self.ev.get_observation())
    def step(self,controls):
        assert len(controls)%10==0
        self.last_observations=[]
        for packet in np.asarray(controls).reshape(-1,10,7):
            for c in packet:
                self.task.data.ctrl[:7]=c
                for _ in range(10):self.task.step_and_log({})
            self.steps+=1
            if self.task.data.warning.number.any():raise RuntimeError("MuJoCo numerical warning")
            hit=bool(self.task.check());self.audit_hits.append(hit)
            self.audit_height.append(float(self.task.object.get_body_pose(self.task.data).pos[2]))
            self.audit_contacts.append([[int(c.geom1),int(c.geom2)] for c in self.task.data.contact])
            if hit and not self.success:self.first_success=float(self.task.data.time)
            self.success|=hit
            self.last_observations.append(copyobs(self.ev.get_observation()))
        return self.last_observations[-1]
    def close(self):self.ev.renderer.close();self.task.finish()
class Native:
    def __init__(self,checkpoint=None):
        from world_model import build,install
        from world_model import load_vae,Diffusion
        from causal_cache import CachedWorldGym
        from world_model import Proprio
        self.cfg=config()["world_model"]
        self.checkpoint=checkpoint or asset("worldgym_task")
        expected=config()["world_model"]["checkpoint_sha256"]
        assert sha(self.checkpoint)==expected,"Native checkpoint changed"
        self.model,self.origin=build(self.cfg)
        install(self.model,torch.load(self.checkpoint,map_location="cpu",weights_only=False)["adapter"])
        self.trainable_names=[n for n,p in self.model.named_parameters() if p.requires_grad]
        self.model.eval()
        self.generator=CachedWorldGym(self.model)
        self.vae,self.vae_origin=load_vae()
        self.diff=Diffusion(sampling_timesteps=10,device="cuda").cuda()
        self.stats=read(asset("action_stats"))
        self.proprio=Proprio(asset("proprio"))
        # Asset hashes and this portable source tree are checked by the entrypoint.
        self.generator.clear()
    def to(self,device):
        self.generator.clear();self.model.to(device);self.vae.to(device)
        if hasattr(self,"latents"):self.latents=self.latents.to(device)
        return self
    def reset(self,initial,seed):
        self.generator.clear();self.model.eval()
        torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
        self.front=initial["observation/image"].copy();self.wrist=initial["observation/wrist_image"].copy()
        self.proprio.reset(initial["observation/state"]);self.current=0;self.actions=[np.zeros(7,np.float32)]
        first=panorama(self.front,self.wrist)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            self.latents=self.vae.encode(torch.from_numpy(first).float().div(255)[None,None].cuda())
        self.last_images=[first]
        return copyobs(initial)
    def step(self,controls):
        from world_model import norm_actions
        controls=np.asarray(controls);assert len(controls)%10==0
        n=len(controls)//10;self.actions.extend(controls[9::10].copy())
        ac=torch.from_numpy(norm_actions(np.asarray(self.actions),self.stats)).unsqueeze(0).cuda()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            self.latents=self.diff.generate(self.generator,self.latents,ac,
                n_context_frames=self.current+1,n_frames=self.current+1+n,horizon=min(5,n),window_len=20,cfg=1.)
            rgb=self.vae.decode(self.latents[:,self.current+1:self.current+1+n])[0].float().clamp(0,1).cpu().numpy()
        self.last_images=(rgb*255).round().astype("uint8")
        self.last_states=[]
        for i,c in enumerate(controls):
            q=self.proprio.control_step(c)
            if i%10==9:self.last_states.append(q.copy())
        self.current+=n;self.front=self.last_images[-1,:,:256].copy();self.wrist=self.last_images[-1,:,256:].copy()
        return {"observation/image":self.front,"observation/wrist_image":self.wrist,"observation/state":q}
    def close(self):
        self.generator.clear()
        del self.model,self.generator,self.vae,self.diff
        gc.collect();torch.cuda.empty_cache()
