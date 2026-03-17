from bert4torch import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/pretrain_ckpt/Qwen/Qwen2-7B-Instruct")


messages = [
    {"role": "user", "content": '你好'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text, '\n')
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt")
print(model_inputs)

output = tokenizer.decode(model_inputs['input_ids'][0].tolist(), skip_special_tokens=False)
print(output)