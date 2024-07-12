'''
基本测试：chatglm3 function call测试
'''
from bert4torch.pipelines import ChatGlm4Cli, ChatGlm4WebGradio, ChatGlm4WebStreamlit, ChatGlm4OpenaiApi
import re

# ===================================参数=======================================
# chatglm3-6b, chatglm3-6b-32k
model_dir = f"/data/pretrain_ckpt/glm/chatglm3-6b"
Chat = ChatGlm4Cli  # cli: 命令行
# Chat = ChatGlm4WebGradio  # gradio: gradio web demo
# Chat = ChatGlm4WebStreamlit  # streamlit: streamlit web demo
# Chat = ChatGlm4OpenaiApi  # openai: openai 接口
# ==============================================================================


generation_config = {
    'topp': 0.8, 
    'temperature': 0.8
    }


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
            tools = tools,  # 是否使用function_call
            generation_config=generation_config)


if __name__ == '__main__':
    demo.run(stream=False)
