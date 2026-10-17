"""Actual Qwen2.5-VL-7B task-domain QLoRA reward. Not an official success label."""
import gc,time
import torch,numpy as np
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
from peft import PeftModel
from control import asset
from data_runtime import TASK
class AdapterReward:
    def __init__(self,device='cpu',adapter=None):
        self.path=asset('qwen7b');self.adapter_path=adapter or asset('reward_adapter')
        self.processor=AutoProcessor.from_pretrained(self.path,local_files_only=True,min_pixels=448*448,max_pixels=448*448)
        self.ids={}
        for word in ['Yes','No']:
            ids=self.processor.tokenizer.encode(word,add_special_tokens=False);assert len(ids)==1;self.ids[word]=ids[0]
        self.model=None;self.location='none';self.offload_seconds=[]
        if device=='cuda':self.to(device)
    def to(self,device):
        if self.model is None:
            if device=='cpu':return self
            quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual'])
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(self.path,local_files_only=True,
                quantization_config=quant,device_map={'':0},torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
            self.model=PeftModel.from_pretrained(self.model,str(self.adapter_path),is_trainable=False).eval().requires_grad_(False)
            self.location='cuda'
        elif self.location!=device:
            t=time.time();self.model.to(device);self.location=device
            gc.collect();torch.cuda.empty_cache();self.offload_seconds.append(dict(device=device,seconds=time.time()-t))
        return self

    def inputs(self,front,wrist,instruction=TASK):
        # Real RGB only. Same fixed caption sanitization for training, calibration, native and physical observations.
        images=[]
        for x in [front,wrist]:
            a=np.asarray(x).copy();a[:int(round(a.shape[0]*32/256))]=0;images.append(Image.fromarray(a))
        question=("Task instruction: "+instruction+" These are simultaneous front and wrist robot camera views. "
          "Is the orange-capped tube currently held by the robot gripper clear above the blue rack? Answer Yes or No.")
        messages=[{"role":"user","content":[{"type":"image"},{"type":"image"},{"type":"text","text":question}]}]
        text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        ins=self.processor(text=text,images=images,return_tensors="pt")
        return {k:v.to(device="cuda",dtype=torch.bfloat16) if v.is_floating_point() else v.cuda() for k,v in ins.items()}

    @torch.inference_mode()
    def score(self,front,wrist,instruction=TASK,caption=False,initial=None):
        t=time.time();ins=self.inputs(front,wrist,instruction)
        logits=self.model(**ins,use_cache=False).logits[0,-1].float()
        two=logits[[self.ids["No"],self.ids["Yes"]]]
        return dict(score=float(two.softmax(-1)[1]),logits=two.tolist(),seconds=time.time()-t,instruction=instruction,
            reward_model="Qwen2.5-VL-7B-Instruct with task-domain QLoRA",mode="direct_task_lora",
            actual_inputs="Current two RGB views and instruction; no simulator labels/state",quantization="NF4 LM / BF16 vision")
