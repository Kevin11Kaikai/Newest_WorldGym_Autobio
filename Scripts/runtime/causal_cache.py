"""Causal WorldGym inference cache. Reuse fixed prefix K/V only inside diffusion sampling."""
import torch
from torch import nn
import torch.nn.functional as F
from world_model import core

class CachedWorldGym(nn.Module):
 def __init__(self,model):super().__init__();self.model=model;self.max_frames=model.max_frames;self.cache={};self.hits=0;self.builds=0
 def get_null_cond(self,a):return self.model.get_null_cond(a)
 def clear(self):self.cache.clear()
 def qkv(self,attention,x):
  B,T,H,W,D=x.shape;v=attention.qkv_proj(x).reshape(B,T,H,W,3,attention.num_heads,D//attention.num_heads).permute(4,0,2,3,5,1,6);return tuple(v.unbind(0))
 def rope(self,q,k,start,total):
  # Same official1D temporal RoPE, positions start..total-1 for queries and keys.
  emb=core.rope_nd((total,),q.shape[-1],rotary_type=core.RotaryType.STANDARD,dtype=q.dtype,device=q.device);emb=emb.view(total,1,2,-1);cos,sin=emb.unbind(-2);cos=cos.reshape(total,-1)[start:];sin=sin.reshape(total,-1)[start:];return core.rope_mix(q,cos,sin),core.rope_mix(k,cos,sin)
 def temporal(self,block,x,c,past=None,total=None,start=0):
  B,T,H,W,D=x.shape;m=block.adaLN_modulation(c).view(B,T,1,1,6,D).unbind(-2);normalized=block.norm1(x)*(1+m[1])+m[0];q,k,v=self.qkv(block.attn,normalized);q,k=self.rope(q,k,start,total or T)
  if past is not None:
   kp,vp=past;k=torch.cat([kp,k],-2);v=torch.cat([vp,v],-2)
   mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(T,device=x.device)[:,None]+start
   out=F.scaled_dot_product_attention(q.reshape(B*H*W,block.attn.num_heads,T,-1),k.reshape(B*H*W,block.attn.num_heads,k.shape[-2],-1),v.reshape(B*H*W,block.attn.num_heads,v.shape[-2],-1),attn_mask=mask if T>1 else None,is_causal=False)
  else:out=F.scaled_dot_product_attention(q.reshape(B*H*W,block.attn.num_heads,T,-1),k.reshape(B*H*W,block.attn.num_heads,T,-1),v.reshape(B*H*W,block.attn.num_heads,T,-1),is_causal=True)
  out=out.reshape(B,H,W,block.attn.num_heads,T,-1).permute(0,4,1,2,3,5).reshape(B,T,H,W,D);out=block.attn.out_proj(out);x=x+out*m[2];x=x+block.ffwd(block.norm2(x)*(1+m[4])+m[3])*m[5];return x,(k,v)
 def build(self,x,t,a):
  z=self.model.patchify(x);c=self.model.get_cond(t,a);kv=[]
  for block in self.model.blocks:
   z=block.s_block(z,c);z,pair=self.temporal(block.t_block,z,c);kv.append(pair)
  self.builds+=1;return kv
 def forward(self,x,t,a):
  assert not self.model.training and not torch.is_grad_enabled(),'Cache is inference only'
  B,T,H,W,C=x.shape
  if B!=1:return self.model(x,t,a)
  context=int((t[0]==14).sum().item())
  if context==0 or context==T or not bool((t[0,:context]==14).all()):return self.model(x,t,a)
  key=bool((a[...,-1]==1).all().item());cached=self.cache.get(key)
  valid=cached is not None and cached['x'].shape==x[:,:context].shape and torch.equal(cached['x'],x[:,:context]) and torch.equal(cached['a'],a[:,:context])
  if not valid:
   kv=self.build(x[:,:context],t[:,:context],a[:,:context]);cached=dict(x=x[:,:context].clone(),a=a[:,:context].clone(),kv=kv);self.cache[key]=cached
  else:self.hits+=1
  z=self.model.patchify(x[:,context:]);c=self.model.get_cond(t[:,context:],a[:,context:])
  for i,block in enumerate(self.model.blocks):z=block.s_block(z,c);z,_=self.temporal(block.t_block,z,c,past=cached['kv'][i],total=T,start=context)
  z=self.model.final_layer(z,c);n=T-context;z=z.reshape(B*n,z.shape[2],z.shape[3],z.shape[4]);z=self.model.unpatchify(z).reshape(B,n,H,W,C)
  # Official DDIM keeps all stabilized context entries via t==t_next; only future v values are consumed.
  return torch.cat([torch.zeros_like(x[:,:context]),z],dim=1)
