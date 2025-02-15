from transformers import AutoModelForCausalLM, AutoTokenizer
from bert4torch.pipelines import Chat
import re


# Yi-6B
# Yi-1.5-9B-Chat-16K
model_dir = "E:/data/pretrain_ckpt/01-ai/Yi-1.5-9B-Chat-16K"


# print('==========================transformers=============================')
# query = "There's a place where time stands still. A place of breath taking wonder, but also"
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
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
generation_config = {
    "include_input": False if re.search('Chat', model_dir) else True
}

demo = Chat(model_dir, 
            generation_config=generation_config,
            )


if __name__ == '__main__':
    demo.run()