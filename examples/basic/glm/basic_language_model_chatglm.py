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
from bert4torch.pipelines import ChatGlmCli, ChatGlmWebGradio, ChatGlmWebStreamlit, ChatGlmOpenaiApi
from bert4torch.pipelines import ChatGlm2Cli, ChatGlm2WebGradio, ChatGlm2WebStreamlit, ChatGlm2OpenaiApi
from bert4torch.pipelines import ChatGlm3Cli, ChatGlm3WebGradio, ChatGlm3WebStreamlit, ChatGlm3OpenaiApi
from bert4torch.pipelines import ChatGlm4Cli, ChatGlm4WebGradio, ChatGlm4WebStreamlit, ChatGlm4OpenaiApi
import re

# ===================================参数=======================================
# chatglm-6b, chatglm-6b-int4, chatglm-6b-int8
# chatglm2-6b, chatglm2-6b-int4, chatglm2-6b-32k
# chatglm3-6b, chatglm3-6b-32k
# glm-4-9b, glm-4-9b-chat, glm-4-9b-chat-1m
model_dir = f"/data/pretrain_ckpt/glm/chatglm3-6b"

# cli: 命令行
# gradio: gradio web demo
# streamlit: streamlit web demo
# openai: openai 接口
mode = 'cli'
# ==============================================================================


generation_config = {
    'topp': 0.8, 
    'temperature': 0.8, 
    'include_input': True if re.search('glm-4-9b$', model_dir) else False, 
    # 'n': 5
    }

ChatMap = {
    'glm-cli': ChatGlmCli,
    'glm-gradio': ChatGlmWebGradio,
    'glm-streamlit': ChatGlmWebStreamlit,
    'glm-openai': ChatGlmOpenaiApi,
    'glm2-cli': ChatGlm2Cli,
    'glm2-gradio': ChatGlm2WebGradio,
    'glm2-streamlit': ChatGlm2WebStreamlit,
    'glm2-openai': ChatGlm2OpenaiApi,
    'glm3-cli': ChatGlm3Cli,
    'glm3-gradio': ChatGlm3WebGradio,
    'glm3-streamlit': ChatGlm3WebStreamlit,
    'glm3-openai': ChatGlm3OpenaiApi,
    'glm4-cli': ChatGlm4Cli,
    'glm4-gradio': ChatGlm4WebGradio,
    'glm4-streamlit': ChatGlm4WebStreamlit,
    'glm4-openai': ChatGlm4OpenaiApi
}

if re.search('glm-4', model_dir):
    Chat = ChatMap[f'glm4-{mode}']
elif re.search('glm3', model_dir):
    Chat = ChatMap[f'glm3-{mode}']
elif re.search('glm2', model_dir):
    Chat = ChatMap[f'glm2-{mode}']
elif re.search('glm', model_dir):
    Chat = ChatMap[f'glm-{mode}']
else:
    raise ValueError('not supported')

tools = [
    {'name': 'track', 'description': '追踪指定股票的实时价格',
     'parameters':
         {
             'type': 'object', 'properties':
             {'symbol':
                 {
                     'description': '需要追踪的股票代码'
                 }
             },
             'required': []
         }
     }, {
        'name': '/text-to-speech', 'description': '将文本转换为语音',
        'parameters':
            {
                'type': 'object', 'properties':
                {
                    'text':
                        {
                            'description': '需要转换成语音的文本'
                        },
                    'voice':
                        {
                            'description': '要使用的语音类型（男声、女声等）'
                        },
                    'speed': {
                        'description': '语音的速度（快、中等、慢等）'
                    }
                }, 'required': []
            }
    },
    {
        'name': '/image_resizer', 'description': '调整图片的大小和尺寸',
        'parameters': {'type': 'object',
                       'properties':
                           {
                               'image_file':
                                   {
                                       'description': '需要调整大小的图片文件'
                                   },
                               'width':
                                   {
                                       'description': '需要调整的宽度值'
                                   },
                               'height':
                                   {
                                       'description': '需要调整的高度值'
                                   }
                           },
                       'required': []
                       }
    },
    {
        'name': '/foodimg', 'description': '通过给定的食品名称生成该食品的图片',
        'parameters': {
            'type': 'object', 'properties':
                {
                    'food_name':
                        {
                            'description': '需要生成图片的食品名称'
                        }
                },
            'required': []
        }
    }
]

demo = Chat(model_dir, 
            # tools = tools,  # 是否使用function_call
            generation_config=generation_config)


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
