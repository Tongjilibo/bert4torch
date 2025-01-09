import torch
from PIL import Image
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
import requests


device = "cuda"
model_dir = 'E:/data/pretrain_ckpt/glm/glm-4v-9b'
url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

query = '描述这张图片'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True).to(device)  # chat mode

model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir, device_map='auto')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
last_len = 0
with torch.no_grad():
    for generated_ids in model.stream_generate(**inputs, **gen_kwargs):
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(output_text[last_len:], flush=True, end='')
        last_len = len(output_text)