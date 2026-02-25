'''
基本测试: chatglm3/glm4 function call测试
'''
from bert4torch.pipelines import Chat

# ===================================参数=======================================
# chatglm3-6b, chatglm3-6b-32k
# glm-4-9b-chat, glm-4-9b-chat-1m
model_dir = f"E:/data/pretrain_ckpt/THUDM/chatglm3-6b"
# ==============================================================================


if 'glm3' in model_dir:
    tools = [
        {
            "name": "track", "description": "追踪指定股票的实时价格",
            "parameters":
                {
                    "type": "object", "properties":
                    {"symbol":
                        {
                            "description": "需要追踪的股票代码"
                        }
                    },
                    "required": []
                }
        }
    ]

else:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        },
    ]


if __name__ == '__main__':
    demo = Chat(model_dir, mode='cli')
    demo.run(functions=tools)
