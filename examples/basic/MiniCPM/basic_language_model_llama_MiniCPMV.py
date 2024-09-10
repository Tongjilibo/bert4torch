from bert4torch.models import build_transformer_model
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

# E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6
model_dir = 'E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6'
model = build_transformer_model(checkpoint_path=model_dir)

model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

image = Image.open('/home/lb/projects/bert4torch/test_local/资料概要.png').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    processor=processor
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
