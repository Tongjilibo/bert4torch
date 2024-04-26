'''
基本测试：chatglm1-3的测试

- 官方项目：https://github.com/THUDM/ChatGLM-6B
- hf链接：https://huggingface.co/THUDM/chatglm-6b
- fp16半精度下显存占用14G

- 官方项目：https://github.com/THUDM/ChatGLM2-6B
- hf链接：https://huggingface.co/THUDM/chatglm2-6b

- 官方项目：https://github.com/THUDM/ChatGLM3-6B
- hf链接：https://huggingface.co/THUDM/chatglm3-6b
- hf链接：https://huggingface.co/THUDM/chatglm3-6b-32k
'''
from bert4torch.pipelines import ChatGlmCli, ChatGlmWebGradio, ChatGlmWebStreamlit, ChatGlmOpenaiApi
from bert4torch.pipelines import ChatGlm2Cli, ChatGlm2WebGradio, ChatGlm2WebStreamlit, ChatGlm2OpenaiApi
from bert4torch.pipelines import ChatGlm3Cli, ChatGlm3WebGradio, ChatGlm3WebStreamlit, ChatGlm3OpenaiApi


# chatglm-6B, chatglm-6B-int4, chatglm-6B-int8 chatglm2-6B, chatglm2-6B-int4, chatglm2-6B-32k, chatglm3-6b, chatglm3-6B-32k
model_dir = f"/data/pretrain_ckpt/glm/chatglm2-6B"

generation_config = {'max_length': 1024, 'topp': 0.8, 'temperature': 0.8}

# demo = ChatGlmCli(model_dir, generation_config=generation_config)
# demo = ChatGlmWebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlmWebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlmOpenaiApi(model_dir, generation_config=generation_config)

demo = ChatGlm2Cli(model_dir, generation_config=generation_config)
# demo = ChatGlm2WebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlm2WebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlm2OpenaiApi(model_dir, generation_config=generation_config)

# demo = ChatGlm3Cli(model_dir, generation_config=generation_config)
# demo = ChatGlm3WebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlm3WebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlm3OpenaiApi(model_dir, generation_config=generation_config)


if __name__ == '__main__':
    if False:
        # 一次性输出N条记录
        demo.generation_config['n'] = 5
        res = demo.chat('如何查询天气？')
        print(res)
    else:
        demo.run()
