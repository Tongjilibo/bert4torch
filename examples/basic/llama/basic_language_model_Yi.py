from transformers import AutoModelForCausalLM, AutoTokenizer
dir_path = "E:/pretrain_ckpt/llama/01-ai@Yi-6B"
query = "There's a place where time stands still. A place of breath taking wonder, but also"

# print('==========================transformers=============================')
# tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(dir_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
# inputs = tokenizer(query, return_tensors="pt")

# outputs = model.generate(
#     inputs.input_ids.cuda(),
#     max_length=256,
#     eos_token_id=tokenizer.eos_token_id,
#     do_sample=True,
#     repetition_penalty=1.3,
#     no_repeat_ngram_size=5,
#     temperature=0.7,
#     top_k=40,
#     top_p=0.8,
# )
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


print('==========================bert4torch=============================')
from bert4torch.models import build_transformer_model
import os
model = build_transformer_model(config_path=dir_path, checkpoint_path=dir_path).half().cuda()

generation_config = {
    "repetition_penalty": 1.3, 
    "temperature": 0.7, 
    "top_k": 40, 
    "top_p": 0.8, 
    "include_input": True
}

response = model.generate(query, **generation_config)
print(response)