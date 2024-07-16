'''
基本测试：chatglm3 function call测试
'''
from bert4torch.pipelines import ChatGlm4Cli, ChatGlm4WebGradio, ChatGlm4WebStreamlit, ChatGlm4OpenaiApi

# ===================================参数=======================================
# glm-4-9b, glm-4-9b-chat, glm-4-9b-chat-1m
model_dir = f"/data/pretrain_ckpt/glm/glm-4-9b-chat"
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

demo = Chat(model_dir, 
            generation_config=generation_config)


if __name__ == '__main__':
    demo.run(functions=tools)
