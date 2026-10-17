"""Task-adapted official WorldGym and learned proprioception. No future simulator access."""
import math,types,importlib.util,sys,random
import numpy as np,torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from diffusers.models import AutoencoderKL
from control import asset,read,sha
def core_module(name):
    path=asset('worldgym_source')/'src/world_model_eval'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('worldgym_core_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module
core=core_module('model');DiT=core.DiT;AttentionType=core.AttentionType;RotaryType=core.RotaryType
Diffusion=core_module('diffusion').Diffusion;VAE=core_module('vae').VAE
MODEL_CONFIG=dict(in_channels=16,patch_size=2,dim=1024,num_layers=16,num_heads=16,action_dim=10,max_frames=20)
CHECKPOINT=asset('worldgym_base')

def load_model():
    ck=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    if isinstance(ck,dict) and 'ema' in ck:state=ck['ema'];branch='ema'
    elif isinstance(ck,dict) and 'model' in ck:state=ck['model'];branch='model'
    else:state=ck;branch='raw_state_dict'
    assert state['x_proj.weight'].shape==(1024,16,2,2)
    assert state['action_embedder.weight'].shape==(1024,10)
    layers=sorted({int(k.split('.')[1]) for k in state if k.startswith('blocks.')})
    assert layers==list(range(16))
    with torch.device('meta'):
        model=DiT(**MODEL_CONFIG,rope_config={AttentionType.SPATIAL:RotaryType.STANDARD,AttentionType.TEMPORAL:RotaryType.STANDARD})
    model.load_state_dict(state,strict=True,assign=True)
    model=model.to('cuda').eval().requires_grad_(False)
    return model,{'load_branch':branch,'strict_load':True,'checkpoint_step':ck.get('step') if isinstance(ck,dict) else None,
        'parameter_count':sum(p.numel() for p in model.parameters()),'configuration':MODEL_CONFIG,'rope':'standard spatial/temporal'}


def load_vae():
    local=asset('vae')
    vae=VAE.__new__(VAE);torch.nn.Module.__init__(vae)
    vae.vae=AutoencoderKL.from_pretrained(str(local),subfolder='vae',local_files_only=True)
    vae.vae.eval().requires_grad_(False)
    return vae.to('cuda'),{'repo_id':'stabilityai/stable-diffusion-3-medium-diffusers','local_directory':str(local)}

class LowRank(nn.Module):
 def __init__(self,base,rank=8,alpha=8):
  super().__init__();self.base=base.requires_grad_(False);self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device));self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device));nn.init.kaiming_uniform_(self.a,a=math.sqrt(5));self.scale=alpha/rank
 def forward(self,x):return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.a),self.b)*self.scale
 def extra_repr(self):return f'rank={self.a.shape[0]}, scale={self.scale}'

def inject(module,rank,alpha):
 for name,child in list(module.named_children()):
  if isinstance(child,nn.Linear):setattr(module,name,LowRank(child,rank,alpha))
  else:inject(child,rank,alpha)

def lora_build(config):
 model,origin=load_model();model.requires_grad_(False)
 for block in model.blocks[-config['last_blocks']:]:inject(block,config['rank'],config['alpha'])
 inject(model.final_layer,config['rank'],config['alpha']);model.action_embedder=nn.Linear(10,1024,device='cuda');nn.init.normal_(model.action_embedder.weight,std=.02);nn.init.zeros_(model.action_embedder.bias)
 # Checkpoint all blocks: native action embedding gradients traverse frozen early blocks.
 for block in model.blocks:
  baseforward=block.forward
  def wrapped(self,x,c,baseforward=baseforward):
   return checkpoint(baseforward,x,c,use_reentrant=False) if self.training and torch.is_grad_enabled() else baseforward(x,c)
  block.forward=types.MethodType(wrapped,block)
 return model,origin

def install(model,state):
 own=model.state_dict();assert set(state)=={k for k,v in model.named_parameters() if v.requires_grad}
 with torch.no_grad():
  for k,v in state.items():own[k].copy_(v.to(own[k].device))


def build(config):
    model,origin=lora_build(config)
    parent=asset('worldgym_parent')
    install(model,torch.load(parent,map_location='cpu',weights_only=False)['adapter'])
    for block in model.blocks[-config['full_weight_last_blocks']:]:block.requires_grad_(True)
    if config['full_weight_final_layer']:model.final_layer.requires_grad_(True)
    return model,dict(origin,parent_adapter_sha256=sha(parent))

class Transition(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(21,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,7))
 def forward(self,x):return self.net(x)

class Proprio:
 def __init__(self,folder):
  self.linear=torch.load(folder/'linear.pt',map_location='cpu',weights_only=True);self.mlp=torch.load(folder/'mlp_best.pt',map_location='cpu',weights_only=True);self.net=Transition();self.net.load_state_dict(self.mlp['state_dict']);self.net.eval();self.q=None;self.dq=None
 def reset(self,q):self.q=torch.as_tensor(q,dtype=torch.float32).reshape(1,7).clone();self.dq=torch.zeros_like(self.q)
 def control_step(self,control):
  c=torch.as_tensor(control,dtype=torch.float32).reshape(1,7);x=torch.cat([self.q,self.dq,c],dim=1);l=self.linear;m=self.mlp
  with torch.inference_mode():
   zl=((x-l['input_mean'])/l['input_std']).clamp(-20,20);zm=((x-m['input_mean'])/m['input_std']).clamp(-20,20);ld=(zl@l['weight']+l['bias'])*l['output_std']+l['output_mean'];md=self.net(zm)*m['output_std']+m['output_mean'];delta=torch.cat([ld[:,:6],md[:,6:]],dim=1);self.q=self.q+delta;self.dq=delta
  assert torch.isfinite(self.q).all(),'Nonfinite autonomous proprio prediction';return self.q[0].numpy().copy()

def norm_actions(a,stats):
 v=np.zeros((len(a),10),np.float32);v[:,:7]=(a-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32);v[0]=0;return v
