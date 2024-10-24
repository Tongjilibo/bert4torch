'''
基本测试：chatglm1-4的测试

- 官方项目：https://github.com/THUDM/ChatGLM-6B
- hf链接：https://huggingface.co/THUDM/chatglm-6b
- fp16半精度下显存占用14G

- 官方项目：https://github.com/THUDM/ChatGLM2-6B
- hf链接：https://huggingface.co/THUDM/chatglm2-6b

- 官方项目：https://github.com/THUDM/ChatGLM3-6B
- hf链接：https://huggingface.co/THUDM/chatglm3-6b
- hf链接：https://huggingface.co/THUDM/chatglm3-6b-32k
'''
from bert4torch.pipelines import Chat
import re

# ===================================参数=======================================
# chatglm-6b, chatglm-6b-int4, chatglm-6b-int8
# chatglm2-6b, chatglm2-6b-int4, chatglm2-6b-32k
# chatglm3-6b, chatglm3-6b-32k
# glm-4-9b, glm-4-9b-chat, glm-4-9b-chat-1m
model_dir = f"E:/data/pretrain_ckpt/glm/chatglm3-6b"

generation_config = {
    'top_p': 0.8, 
    'temperature': 0.8, 
    'include_input': True if re.search('glm-4-9b$', model_dir) else False, 
    # 'n': 5
    }

# cli: 命令行
# gradio: gradio web demo
# streamlit: streamlit web demo
# openai: openai 接口
demo = Chat(model_dir, generation_config=generation_config, mode='cli')


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
        # 聊天
        demo.run()
