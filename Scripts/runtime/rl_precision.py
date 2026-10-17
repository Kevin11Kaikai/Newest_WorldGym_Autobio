"""FP32 trainable VLA parameters for GRPO, with BF16 autocast and a frozen BF16 VLM."""
from policy_runtime import Actor,fingerprint
from control import H,read
import torch
class RLActor(Actor):
    def __init__(self,ckpt=None,resolution=256):
        super().__init__(ckpt,resolution)
        self.set_rl()
        for p in self.parameters():p.data=p.data.float()
        if ckpt:
            # Actor's original constructor creates a BF16 expert. Reload after promotion so
            # saved FP32 RL updates are never rounded through BF16 during checkpoint recovery.
            c=torch.load(ckpt,map_location="cpu",weights_only=False)
            missing,extra=self.model.load_state_dict(c["adapter"],strict=False)
            assert not extra and all(k.startswith("vlm_with_expert.vlm.") for k in missing)
            assert fingerprint(self.adapter())==fingerprint(c["adapter"]),"FP32 checkpoint changed during loading"
        assert all(p.dtype==torch.float32 for p in self.parameters())
