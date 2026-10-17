from collections import OrderedDict
import numpy as np,torch
from torch.utils.data import Dataset
from control import asset
CACHE=asset('decoded')
class EpisodeDataset(Dataset):
    def __init__(self,episode_ids,anchors=None):
        self.episode_ids=list(episode_ids);self.numeric={};self.offsets=[0];self.image_cache=OrderedDict()
        for ep in episode_ids:
            data=np.load(CACHE/f'episode_{ep:06d}_numeric.npz')
            self.numeric[ep]={'state':data['state'],'actions':data['actions']}
            self.offsets.append(self.offsets[-1]+len(data['state'])-1)
        self.anchors=anchors
    def __len__(self):return len(self.anchors) if self.anchors is not None else self.offsets[-1]
    def locate(self,idx):
        if self.anchors is not None:return self.anchors[idx]
        pos=int(np.searchsorted(self.offsets,idx,side='right')-1)
        return self.episode_ids[pos],idx-self.offsets[pos]
    def __getitem__(self,idx):
        ep,frame=self.locate(idx);data=self.numeric[ep];n=len(data['state'])
        future=frame+1+np.arange(50);pad=future>=n;future=np.minimum(future,n-1)
        batch={'observation.state':torch.from_numpy(data['state'][frame].copy()),'action':torch.from_numpy(data['actions'][future].copy()),'action_is_pad':torch.from_numpy(pad)}
        for camera,key in [('image','observation.images.front'),('wrist_image','observation.images.wrist')]:
            ck=(ep,camera)
            if ck not in self.image_cache:self.image_cache[ck]=np.load(CACHE/f'episode_{ep:06d}_{camera}.npy',mmap_mode='r')
            self.image_cache.move_to_end(ck)
            if len(self.image_cache)>16:self.image_cache.popitem(last=False)
            batch[key]=torch.from_numpy(self.image_cache[ck][frame].copy()).permute(2,0,1).float()/255
        return batch

def to_device(batch,device='cuda'):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

def close_onsets(ep):
    a=np.load(CACHE/f'episode_{ep:06d}_numeric.npz')['actions']
    closed=np.isclose(a[:,6],.013,atol=1e-5)
    return np.flatnonzero(closed&np.r_[True,~closed[:-1]])

def event_indices(dataset):
    indices=[]
    for k,ep in enumerate(dataset.episode_ids):
        n=len(dataset.numeric[ep]['state'])
        for onset in close_onsets(ep):
            lo=max(0,int(onset)-50);hi=min(n-2,int(onset)+10)
            indices.extend(dataset.offsets[k]+f for f in range(lo,hi+1))
    return np.unique(indices)

def validation_anchors(ids):
    anchors=[]
    for ep in ids:
        n=len(np.load(CACHE/f'episode_{ep:06d}_numeric.npz')['state'])
        frames=set(np.linspace(0,n-2,32,dtype=int).tolist())
        for onset in close_onsets(ep):
            for rel in [-50,-20,-10,-5,-2,-1,0,1,2,5,10,30,60]:
                frames.add(int(np.clip(onset+rel,0,n-2)))
        anchors.extend((ep,f) for f in sorted(frames))
    return anchors

def augment(batch,settings):
    if not settings:return batch
    result=dict(batch)
    for name in ['observation.images.front','observation.images.wrist']:
        image=batch[name];n=image.shape[0]
        brightness=1+(torch.rand(n,1,1,1,device=image.device)*2-1)*settings['brightness']
        contrast=1+(torch.rand(n,1,1,1,device=image.device)*2-1)*settings['contrast']
        mean=image.mean(dim=(-2,-1),keepdim=True)
        result[name]=((image-mean)*contrast+mean).mul(brightness).clamp(0,1)
    return result
