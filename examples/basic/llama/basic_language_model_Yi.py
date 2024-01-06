from transformers import AutoModelForCausalLM, AutoTokenizer
dir_path = "E:/pretrain_ckpt/llama/01-ai@Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
max_length = 256
query = "There's a place where time stands still. A place of breath taking wonder, but also"

# print('==========================transformers=============================')
# model = AutoModelForCausalLM.from_pretrained(dir_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
# inputs = tokenizer(query, return_tensors="pt")

# outputs = model.generate(
#     inputs.input_ids.cuda(),
#     max_length=max_length,
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
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half().cuda()

generation_config = {
    "tokenizer": tokenizer, 
    "start_id": None, 
    "end_id": tokenizer.eos_token_id, 
    "mode":'random_sample', 
    "tokenizer_decode_config": {'skip_special_tokens': True},
    "max_length": max_length, 
    "default_rtype": 'logits', 
    "use_states": True,
    "repetition_penalty": 1.3, 
    "no_repeat_ngram_size": 5,
    "temperature": 0.7, 
    "top_k": 40, 
    "top_p": 0.8, 
    "include_input": True
}

response = model.generate(query, **generation_config)
print(response)