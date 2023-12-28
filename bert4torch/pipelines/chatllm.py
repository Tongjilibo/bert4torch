'''
该模块的主要功能有两个
1. 很多chat大模型有build_prompt操作，有的操作较为复杂，这里预制以减轻代码重复
2. 提供CliDemo, WebDemo, OpenApiDemo用于快速搭建demo
'''

import os
import torch
import re
from bert4torch.models import build_transformer_model
from bert4torch.snippets import log_warn_once, cuda_empty_cache


class Chat:
    '''聊天类
    :param model_path: str, 模型权重地址，可以是所在文件夹、文件地址、文件地址列表
    :param half: bool, 是否半精度
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quantization_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'maxlen':2048, 'default_rtype':'logits', 'use_states':True}
    '''
    def __init__(self, model_path, half=True, quantization_config=None, generation_config=None, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.checkpoint_path = model_path
        self.config_path = os.path.join(model_path, 'bert4torch_config.json')
        self.generation_config = generation_config if generation_config is not None else kwargs
        self.half = half
        self.quantization_config = quantization_config
        self.tokenizer = self.build_tokenizer()
        self.generation_config['tokenizer'] = self.tokenizer
        self.model = self.build_model()

    def build_prompt(self, query, history) -> str:
        '''对query和history进行处理，生成进入模型的text
        :param query: str, 最近的一次user的input
        :param history: List, 历史对话记录，格式为[(input1, response1), (input2, response2)]
        '''
        raise NotImplementedError
    
    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def build_model(self):
        model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path)
        # 半精度
        if self.half:
            model = model.half()
        # 量化
        if self.quantization_config is not None:
            model = model.quantize(**self.quantization_config)
        return model.to(self.device)
    
    def process_response(self, response, history=None):
        '''对response进行后处理，可自行继承后来自定义'''
        return response

    def generate(self, query:str, history=[]):
        prompt = self.build_prompt(query, history)
        return self.model.generate(prompt, **self.generation_config)

    def batch_generate(self, query:list, history=[]):
        prompts = [self.build_prompt(q, history) for q in query]
        return self.model.batch_generate(prompts, **self.generation_config)

    def stream_generate(self, query:str, history=[]):
        '''单条样本stream输出预测的结果'''
        prompt = self.build_prompt(query, history)
        for response in self.model.stream_generate(prompt, **self.generation_config):
            yield response


class ChatCli(Chat):
    '''在命令行中交互的demo'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_str = "输入内容进行对话，clear清空对话历史，stop终止程序"
        self.history_maxlen = 3

    def build_cli_text(self, history):
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for query, response in history:
            prompt += f"\n\nUser：{query}"
            prompt += f"\n\nAssistant：{response}"
        return prompt

    def run(self, stream=True):
        import platform
        os_name = platform.system()
        previous_history, history = [], []
        clear_command = 'cls' if os_name == 'Windows' else 'clear'
        print(self.init_str)
        while True:
            query = input("\nUser: ")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                previous_history, history = [], []
                os.system(clear_command)
                print(self.init_str)
                continue
            
            prompt = self.build_prompt(query, history)
            if stream:
                for response in self.model.stream_generate(prompt, **self.generation_config):
                    response = self.process_response(response, history)
                    new_history = history + [(query, response)]
                    os.system(clear_command)
                    print(self.build_cli_text(previous_history + new_history), flush=True)
            else:
                response = self.model.generate(prompt, **self.generation_config)
                response = self.process_response(response, history)
                new_history = history + [(query, response)]
            
            os.system(clear_command)
            print(self.build_cli_text(previous_history + new_history), flush=True)
            history = new_history[-self.history_maxlen:]
            if len(new_history) > self.history_maxlen:
                previous_history += new_history[:-self.history_maxlen]
            cuda_empty_cache()


class ChatWeb(Chat):
    '''gradio实现的网页交互的demo
    默认是stream输出，默认history不会删除，需手动清理
    '''
    def __init__(self, *args, max_length=4096, **kwargs):
        super().__init__(*args, **kwargs)
        import gradio as gr
        self.gr = gr
        self.max_length = max_length
        self.max_repetition_penalty = 10
        self.stream = True  # 一般都是流式，因此未放在页面配置项
        log_warn_once('`gradio` changes frequently, the code is successfully tested under 3.44.4')

    def reset_user_input(self):
        return self.gr.update(value='')

    @staticmethod
    def reset_state():
        return [], []

    def set_generation_config(self, max_length, top_p, temperature, repetition_penalty):
        '''根据web界面的参数修改生成参数'''
        self.generation_config['max_length'] = max_length
        self.generation_config['top_p'] = top_p
        self.generation_config['temperature'] = temperature
        self.generation_config['repetition_penalty'] = repetition_penalty

    def __stream_predict(self, input, chatbot, history, max_length, top_p, temperature, repetition_penalty):
        '''流式生成'''
        self.set_generation_config(max_length, top_p, temperature, repetition_penalty)
        chatbot.append((input, ""))
        input_text = self.build_prompt(input, history)
        for response in self.model.stream_generate(input_text, **self.generation_config):
            response = self.process_response(response, history)
            chatbot[-1] = (input, response)
            new_history = history + [(input, response)]
            yield chatbot, new_history
        cuda_empty_cache()  # 清理显存

    def __predict(self, input, chatbot, history, max_length, top_p, temperature, repetition_penalty):
        '''一次性生成'''
        self.set_generation_config(max_length, top_p, temperature, repetition_penalty)
        chatbot.append((input, ""))
        input_text = self.build_prompt(input, history)
        response = self.model.generate(input_text, **self.generation_config)
        response = self.process_response(response, history)
        chatbot[-1] = (input, response)
        new_history = history + [(input, response)]
        cuda_empty_cache()  # 清理显存
        return chatbot, new_history

    def run(self, **launch_configs):
        with self.gr.Blocks() as demo:
            self.gr.HTML("""<h1 align="center">Chabot Web Demo</h1>""")

            chatbot = self.gr.Chatbot()
            with self.gr.Row():
                with self.gr.Column(scale=4):
                    with self.gr.Column(scale=12):
                        user_input = self.gr.Textbox(show_label=False, placeholder="Input...", lines=10) # .style(container=False)
                    with self.gr.Column(min_width=32, scale=1):
                        submitBtn = self.gr.Button("Submit", variant="primary")
                with self.gr.Column(scale=1):
                    emptyBtn = self.gr.Button("Clear History")
                    max_length = self.gr.Slider(0, self.max_length, value=self.max_length//2, step=1.0, label="Maximum length", interactive=True)
                    top_p = self.gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature = self.gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                    repetition_penalty = self.gr.Slider(0, self.max_repetition_penalty, value=1, step=0.1, label="Repetition penalty", interactive=True)

            history = self.gr.State([])
            if self.stream:
                submitBtn.click(self.__stream_predict, [user_input, chatbot, history, max_length, top_p, temperature, repetition_penalty], [chatbot, history], show_progress=True)
            else:
                submitBtn.click(self.__predict, [user_input, chatbot, history, max_length, top_p, temperature, repetition_penalty], [chatbot, history], show_progress=True)

            submitBtn.click(self.reset_user_input, [], [user_input])
            emptyBtn.click(self.reset_state, outputs=[chatbot, history], show_progress=True)

        demo.queue().launch(**launch_configs)


def extend_with_cli(InputModel):
    """添加ChatCliDemo"""
    class ChatDemo(InputModel, ChatCli):
        pass
    return ChatDemo


def extend_with_web(InputModel):
    """添加ChatWebDemo"""
    class ChatDemo(InputModel, ChatWeb):
        pass
    return ChatDemo


# Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

import time
import json
import requests
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from bert4torch.snippets import log_info, log_warn, cuda_empty_cache, AnyClass
from bert4torch.snippets import is_fastapi_available, is_pydantic_available, is_sseclient_available
from packaging import version

FastAPI, BaseModel, Field= object, object, AnyClass
if is_fastapi_available():
    from fastapi import FastAPI, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
if is_pydantic_available():
    from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    cuda_empty_cache()


class _ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class _ModelList(BaseModel):
    object: str = "list"
    data: List[_ModelCard] = []


class _ChatMessage(BaseModel):
    role: str
    content: str


class _DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class _ChatCompletionRequest(BaseModel):
    model: str
    messages: List[_ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class _ChatCompletionResponseChoice(BaseModel):
    index: int
    message: _ChatMessage
    finish_reason: Literal["stop", "length"]


class _ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: _DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class _ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[_ChatCompletionResponseChoice, _ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


class ChatOpenaiApi(Chat):
    """OpenAi的api

    :param model_path: str, 模型所在的文件夹地址
    :param name: str, 模型名称
    :param generation_config: dict, 模型generate的参数设置
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    """
    def __init__(self, model_path, name='default_model', route_api='/chat', route_models='/models', **kwargs):
        super().__init__(model_path, **kwargs)
        assert is_fastapi_available(), "No module found, use `pip install fastapi`"
        from sse_starlette.sse import ServerSentEvent, EventSourceResponse
        import sse_starlette
        if version.parse(sse_starlette.__version__) > version.parse('1.8'):
            log_warn('Module `sse_starlette` above 1.8 not support stream output')
        self.EventSourceResponse = EventSourceResponse
        self.name = name
        self.role_user = 'user'
        self.role_assistant = 'assistant'
        self.role_system = 'system'
        self.app = FastAPI(lifespan=lifespan)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 添加路由
        router = APIRouter()
        router.add_api_route(route_models, methods=['GET'], endpoint=self.list_models, response_model=_ModelList)
        router.add_api_route(route_api, methods=['POST'], endpoint=self.create_chat_completion, response_model=_ChatCompletionResponse)
        self.app.include_router(router)

        log_info('''The request post format should be 
            {
                "messages": [
                    {"content": "你好", "role": "user"},
                    {"content": "你好，我是法律大模型", "role": "assistant"},
                    {"content": "基金从业可以购买股票吗", "role": "user"}
                    ],
                "model": "default",
                "stream": True
            }
            ''')
        
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)

    async def list_models(self):
        model_card = _ModelCard(id=self.name)
        return _ModelList(data=[model_card])

    async def create_chat_completion(self, request: _ChatCompletionRequest):
        if request.messages[-1].role != self.role_user:
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == self.role_system:
            query = prev_messages.pop(0).content + query

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == self.role_user and prev_messages[i+1].role == self.role_assistant:
                    history.append((prev_messages[i].content, prev_messages[i+1].content))
                else:
                    raise HTTPException(status_code=400, detail=f'Arg `messages` do not follow {self.role_user}, \
                                        {self.role_assistant} format.')
        else:
            log_warn(f'prev_messages={len(prev_messages)}%2 != 0, use current query without history instead.')
        
        input_text = self.build_prompt(query, history)

        # 流式输出
        if request.stream:
            generate = self.predict(input_text, request.model)
            return self.EventSourceResponse(generate, media_type="text/event-stream")
        
        # 非流式输出
        else:
            response = self.model.generate(input_text, **self.generation_config)
            choice_data = _ChatCompletionResponseChoice(
                index=0,
                message=_ChatMessage(role=self.role_assistant, content=response),
                finish_reason="stop"
            )

            return _ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, query: str, model_id: str):
        choice_data = _ChatCompletionResponseStreamChoice(
            index=0,
            delta=_DeltaMessage(role=self.role_assistant),
            finish_reason=None
        )
        chunk = _ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        current_length = 0

        for new_response in self.model.stream_generate(query, **self.generation_config):
            if len(new_response) == current_length:
                continue

            new_text = new_response[current_length:]
            current_length = len(new_response)

            choice_data = _ChatCompletionResponseStreamChoice(
                index=0,
                delta=_DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = _ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))


        choice_data = _ChatCompletionResponseStreamChoice(
            index=0,
            delta=_DeltaMessage(),
            finish_reason="stop"
        )
        chunk = _ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        yield '[DONE]'


class ChatOpenaiClient:
    '''调用openai接口的client, 流式请求
    '''
    def __init__(self, url) -> None:
        self.url = url
        if is_sseclient_available():
            import sseclient
        else:
            raise ImportError('No module found, you may `pip install sseclient-py`')
        
        self.sseclient = sseclient
        log_info('''The body format should be 
            {
                "messages": [
                    {"content": "你好", "role": "user"},
                    {"content": "你好，我是法律大模型", "role": "assistant"},
                    {"content": "基金从业可以购买股票吗", "role": "user"}
                    ],
                "model": "default",
                "stream": True
            }
            ''')
   
    def post(self, body):
        '''接口调用'''
        reqHeaders = {'Accept': 'text/event-stream'}
        request = requests.post(self.url, stream=True, headers=reqHeaders, json=body)
        client = self.sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    yield data['content']

    def post_test(self, body):
        '''简单测试在命令行打印'''
        reqHeaders = {'Accept': 'text/event-stream'}
        request = requests.post(self.url, stream=True, headers=reqHeaders, json=body)
        client = self.sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    print(data['content'], end="", flush=True)


def extend_with_chat_openai_api(InputModel):
    """添加ChatWebDemo"""
    class ChatDemo(InputModel, ChatOpenaiApi):
        pass
    return ChatDemo


class ChatGlm(Chat):
    def build_prompt(self, query, history) -> str:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt
    
    def process_response(self, response, *args):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
ChatGlmCli = extend_with_cli(ChatGlm)
ChatGlmWeb = extend_with_web(ChatGlm)
ChatGlmOpenaiApi = extend_with_chat_openai_api(ChatGlm)


class ChatGlm2(Chat):
    def build_prompt(self, query, history=[]):
        # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i+1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)
        return prompt
    
    def process_response(self, response, *args):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
ChatGlm2Cli = extend_with_cli(ChatGlm2)
ChatGlm2Web = extend_with_web(ChatGlm2)
ChatGlm2OpenaiApi = extend_with_chat_openai_api(ChatGlm2)


class ChatGlm3(Chat):
    def build_prompt(self, query, history=[]):
        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        if (len(history) > 0) and isinstance(history[-1], tuple):
            history.pop()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})
        input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
        return input_ids

    def build_cli_text(self, history):
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for hist in history[:-1]:  # 去除ChatCliDemo添加的当前回复的记录
            if hist['role'] == 'user':
                query = hist['content']
                prompt += f"\n\nUser：{query}"
            elif hist['role'] == 'assistant':
                response = hist['content']
                prompt += f"\n\nAssistant：{response}"
        return prompt
    
    def process_response(self, response, history):
        if (not response) or (response[-1] == "�"):
            return response, history

        content = ""
        for response in response.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content
ChatGlm3Cli = extend_with_cli(ChatGlm3)
ChatGlm3Web = extend_with_web(ChatGlm3)
ChatGlm3OpenaiApi = extend_with_chat_openai_api(ChatGlm3)


class ChatInternLM(Chat):
    def build_prompt(self, query, history=[]):
        prompt = ""
        for user, bot in history:
            prompt += f"""<s><|User|>:{user}<eoh>\n<|Bot|>:{bot}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        if query is not None:
            prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        
        return prompt

    def process_response(self, response, history=None):
        for reg in ['<s>', '</s>', '<eoh>', '<eoa>']:
            response = response.replace(reg, '')
        return response
ChatInternLMCli = extend_with_cli(ChatInternLM)
ChatInternLMWeb = extend_with_web(ChatInternLM)
ChatInternLMOpenaiApi = extend_with_chat_openai_api(ChatInternLM)


class ChatQwen(Chat):
    def __init__(self, *args, system='', max_window_size=6144, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.max_window_size = max_window_size

    def build_prompt(self, query, history) -> str:
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", self.system)
        raw_text = ""

        for turn_query, turn_response in reversed(history):
            query_text = _tokenize_str("user", turn_query)
            response_text = _tokenize_str("assistant", turn_response)
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = len(self.tokenizer.encode(raw_text, allowed_special={im_start, im_end}))
            if current_context_size < self.max_window_size:
                raw_text = prev_chat + raw_text
            else:
                break

        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text
ChatQwenCli = extend_with_cli(ChatQwen)
ChatQwenWeb = extend_with_web(ChatQwen)
ChatQwenOpenaiApi = extend_with_chat_openai_api(ChatQwen)


class ChatLLaMA2(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        if system is None:
            self.system = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
        else:
            self.system = system

    def build_prompt(self, query, history) -> str:
        texts = [f'[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n']
        for user_input, response in history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        texts.append(f'{query.strip()} [/INST]')
        return ''.join(texts)
ChatLLaMA2Cli = extend_with_cli(ChatLLaMA2)
ChatLLaMA2Web = extend_with_web(ChatLLaMA2)
ChatLLaMA2OpenaiApi = extend_with_chat_openai_api(ChatLLaMA2)


class ChatZiya(Chat):
    def build_prompt(self, query, history) -> str:
        prompt = ''
        for human, bot in history:
            prompt += f"<human>:{human}\n<bot>:{bot}\n"
        prompt += f"<human>:{query.strip()}\n<bot>:"
        return prompt
ChatZiyaCli = extend_with_cli(ChatZiya)
ChatZiyaWeb = extend_with_web(ChatZiya)
ChatZiyaOpenaiApi = extend_with_chat_openai_api(ChatZiya)


class ChatChineseAlphaLLaMA(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        if system is None:
            self.system = \
("Below is an instruction that describes a task. "
"Write a response that appropriately completes the request.\n\n"
)
        else:
            self.system = system

    def build_prompt(self, query, history) -> str:
        prompt = ''
        for inst, resp in history:
            prompt += f"### Instruction:\n\n{inst}\n\n### Response:\n\n{resp}\n\n"
        prompt += f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        prompt = self.system +prompt
        return prompt
ChatChineseAlphaLLaMACli = extend_with_cli(ChatChineseAlphaLLaMA)
ChatChineseAlphaLLaMAWeb = extend_with_web(ChatChineseAlphaLLaMA)
ChatChineseAlphaLLaMAOpenaiApi = extend_with_chat_openai_api(ChatChineseAlphaLLaMA)


class ChatBelle(Chat):
    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
    
    def build_prompt(self, query, history) -> str:
        prompt = ''
        for item in history:
            prompt += f"Human: {item[0]} \n\nAssistant: {item[1]}\n\n"
        prompt += f"Human: {query} \n\nAssistant: "
        return prompt
ChatBelleCli = extend_with_cli(ChatBelle)
ChatBelleWeb = extend_with_web(ChatBelle)
ChatBelleOpenaiApi = extend_with_chat_openai_api(ChatBelle)


class ChatBaichuan(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_token_id = 195
        self.assistant_token_id = 196

    def build_prompt(self, query, history) -> str:
        total_input = []
        for user, assistant in history:
            total_input += [self.user_token_id] + self.tokenizer.encode(user)  
            total_input += [self.assistant_token_id] + self.tokenizer.encode(assistant) + [self.tokenizer.eos_token_id]
        total_input += [self.user_token_id] + self.tokenizer.encode(query)
        total_input.append(self.assistant_token_id)
        return total_input
ChatBaichuanCli = extend_with_cli(ChatBaichuan)
ChatBaichuanWeb = extend_with_web(ChatBaichuan)
ChatBaichuanOpenaiApi = extend_with_chat_openai_api(ChatBaichuan)
