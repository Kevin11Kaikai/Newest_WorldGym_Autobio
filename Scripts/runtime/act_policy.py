"""ACT comparison adapter: a full 50-control chunk, with full-precision inference."""
import torch,numpy as np,cv2
from train_act import load_act
from control import sha
class ACTActor:
    def __init__(self,checkpoint):
        self.policy=load_act(checkpoint).eval().requires_grad_(False)
        self.run_fingerprint=sha(checkpoint/'model.safetensors')
    def mean(self,b):
        with torch.autocast('cuda',enabled=False):return self.policy.predict_action_chunk(b).float()
    def obs_batch(self,o,instruction=None):
        b={'observation.state':torch.tensor(o['observation/state'],dtype=torch.float32,device='cuda')[None]}
        for src,key in [('observation/image','front'),('observation/wrist_image','wrist')]:
            b['observation.images.'+key]=torch.from_numpy(cv2.resize(o[src],(224,224)).copy()).permute(2,0,1)[None].float().cuda()/255
        return b
    def physical(self,a):return a.detach().cpu().numpy()
