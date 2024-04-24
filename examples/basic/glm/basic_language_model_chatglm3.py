#! -*- coding: utf-8 -*-
# 基本测试：chatglm3的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM3-6B
# hf链接：https://huggingface.co/THUDM/chatglm3-6b
# hf链接：https://huggingface.co/THUDM/chatglm3-6b-32k


from bert4torch.pipelines import ChatGlm3Cli
from transformers import AutoTokenizer


model_path = "E:/pretrain_ckpt/glm/chatglm3-6b"
# model_path = "E:/pretrain_ckpt/glm/chatglm3-6B-32k"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
generation_kwargs = {"max_length": 2048, 
              "topk": 50, 
              "topp": 0.7, 
              "temperature": 0.95,
              "start_id": None,
              "end_id": [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), 
                         tokenizer.get_command("<|observation|>")],
              "mode": 'random_sample',
              "default_rtype": 'logits',
              "use_states": True}
demo = ChatGlm3Cli(model_path, **generation_kwargs)


if __name__ == '__main__':
    demo.run()