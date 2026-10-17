"""Reproduce one ACT supervised fine-tuning arm from the frozen 10-demo ACT baseline."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from cli import common,begin,gpu_lock,finish
from control import asset,sha,write
def main():
    ap=argparse.ArgumentParser(description=__doc__);common(ap)
    ap.add_argument('--condition',type=int,choices=[10,20,40],default=40)
    ap.add_argument('--seed',type=int,choices=[5101,5102],default=5101)
    ap.add_argument('--smoke',action='store_true',help='Two real optimizer steps and exact checkpoint reload check; not a performance result.')
    args=ap.parse_args()
    with gpu_lock():
        cfg=begin(args,'act_sft',dict(command='act_sft',condition=args.condition,seed=args.seed,
            purpose='SMOKE' if args.smoke else 'REPRODUCTION'))
        p=dict(sft=cfg['act_sft'],base_act=dict(directory=str(asset('act_base')),checkpoint_sha256=sha(asset('act_base')/'model.safetensors')))
        if not (args.run/'protocol.json').exists():write(args.run/'protocol.json',p)
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
        from train_act import run
        finish(args.run,run(args.condition,args.seed,p,args.smoke))
if __name__=='__main__':main()
