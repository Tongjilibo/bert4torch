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
from bert4torch.pipelines import ChatGlm4Cli, ChatGlm4WebGradio, ChatGlm4WebStreamlit, ChatGlm4OpenaiApi
import re


# chatglm-6B, chatglm-6B-int4, chatglm-6B-int8
# chatglm2-6B, chatglm2-6B-int4, chatglm2-6B-32k
# chatglm3-6b, chatglm3-6B-32k
# glm-4-9b, glm-4-9b-chat
model_dir = f"/data/pretrain_ckpt/glm/glm-4-9b"

generation_config = {
    'max_length': 1024, 
    'topp': 0.8, 
    'temperature': 0.8, 
    'include_input': True if re.search('glm-4-9b$', model_dir) else False,
    # 'n': 5
    }

# demo = ChatGlmCli(model_dir, generation_config=generation_config)
# demo = ChatGlmWebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlmWebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlmOpenaiApi(model_dir, generation_config=generation_config)

# demo = ChatGlm2Cli(model_dir, generation_config=generation_config)
# demo = ChatGlm2WebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlm2WebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlm2OpenaiApi(model_dir, generation_config=generation_config)

# demo = ChatGlm3Cli(model_dir, generation_config=generation_config)
# demo = ChatGlm3WebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlm3WebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlm3OpenaiApi(model_dir, generation_config=generation_config)

demo = ChatGlm4Cli(model_dir, generation_config=generation_config)
# demo = ChatGlm4WebGradio(model_dir, generation_config=generation_config)
# demo = ChatGlm4WebStreamlit(model_dir, generation_config=generation_config)
# demo = ChatGlm4OpenaiApi(model_dir, generation_config=generation_config)


if __name__ == '__main__':
    if generation_config.get('n') is not None:
        # 一次性输出N条记录
        res = demo.chat('如何查询天气？')
        print(res)

    elif generation_config.get('include_input', False):
        # 命令行续写
        while True:
            query = input('\n输入:')
            response = demo.generate(query)
            print(f'续写: {response}')

    else:
        # 命令行聊天
        demo.run()
