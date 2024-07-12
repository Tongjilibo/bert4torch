''' 大模型聊天的pipeline调用
主要功能：
1. 命令行调用各个模型demo
2. 利用fastapi为大模型搭建openai格式的server和client调用
    Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
    Usage: python openai_api.py
    Visit http://localhost:8000/docs for documents.
3. web界面快速搭建demo

# TODO: 设置return_states=True时候，受到build_prompt影响，很难保证prompt完全复现
这里采用添加self.generation_config['states']['last_token']，是因为推理完成可能是因为到达max_length，未必是遇到了eos
'''

import os
import torch
from typing import Union, Optional, List, Tuple, Literal, Dict
from bert4torch.models import build_transformer_model
from bert4torch.snippets import (
    log_warn_once, 
    get_config_path, 
    log_info, 
    log_warn, 
    cuda_empty_cache,
    is_fastapi_available, 
    is_pydantic_available, 
    is_sseclient_available, 
    is_streamlit_available
)
from packaging import version
import gc
import time
import json
import requests
from contextlib import asynccontextmanager
import threading
import re


if is_fastapi_available():
    from fastapi import FastAPI, HTTPException, APIRouter, Depends
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.middleware.cors import CORSMiddleware
else:
    class FastAPI: pass
    class HTTPAuthorizationCredentials: pass
    Depends, HTTPBearer = object, object

if is_pydantic_available():
    from pydantic import BaseModel, Field
else:
    BaseModel, Field = object, object

if is_streamlit_available():
    import streamlit as st
else:
    # 防止streamlit不存在时候报错
    import bert4torch.snippets as st
    st.cache_resource = st.delete_arguments


__all__ = [
    'ChatOpenaiApi',
    'ChatOpenaiClient',
    'ChatOpenaiClientSseclient',
    'ChatOpenaiApi',
    'ChatOpenaiClient',
    'ChatOpenaiClientSseclient',
    'ChatGlm',
    'ChatGlmCli',
    'ChatGlmWebGradio',
    'ChatGlmWebStreamlit',
    'ChatGlmOpenaiApi',
    'ChatGlm2',
    'ChatGlm2Cli',
    'ChatGlm2WebGradio',
    'ChatGlm2WebStreamlit',
    'ChatGlm2OpenaiApi',
    'ChatGlm3',
    'ChatGlm3Cli',
    'ChatGlm3WebGradio',
    'ChatGlm3WebStreamlit',
    'ChatGlm3OpenaiApi',
    'ChatGlm4',
    'ChatGlm4Cli',
    'ChatGlm4WebGradio',
    'ChatGlm4WebStreamlit',
    'ChatGlm4OpenaiApi',
    'ChatInternLM',
    'ChatInternLMCli',
    'ChatInternLMWebGradio',
    'ChatInternLMWebStreamlit',
    'ChatInternLMOpenaiApi',
    'ChatQwen',
    'ChatQwenCli',
    'ChatQwenWebGradio',
    'ChatQwenWebStreamlit',
    'ChatQwenOpenaiApi',
    'ChatLLaMA2',
    'ChatLLaMA2Cli',
    'ChatLLaMA2WebGradio',
    'ChatLLaMA2WebStreamlit',
    'ChatLLaMA2OpenaiApi',
    'ChatLLaMA3',
    'ChatLLaMA3Cli',
    'ChatLLaMA3WebGradio',
    'ChatLLaMA3WebStreamlit',
    'ChatLLaMA3OpenaiApi',
    'ChatZiya',
    'ChatZiyaCli',
    'ChatZiyaWebGradio',
    'ChatZiyaWebStreamlit',
    'ChatZiyaOpenaiApi',
    'ChatChineseAlphaLLaMA',
    'ChatChineseAlphaLLaMACli',
    'ChatChineseAlphaLLaMAWebGradio',
    'ChatChineseAlphaLLaMAWebStreamlit',
    'ChatChineseAlphaLLaMAOpenaiApi',
    'ChatBelle',
    'ChatBelleCli',
    'ChatBelleWebGradio',
    'ChatBelleWebStreamlit',
    'ChatBelleOpenaiApi',
    'ChatBaichuan',
    'ChatBaichuanCli',
    'ChatBaichuanWebGradio',
    'ChatBaichuanWebStreamlit',
    'ChatBaichuanOpenaiApi'
    ]


# 一些通用的system话术
SYSTEM_ZH = """你是一个乐于助人、尊重他人、诚实的中文聊天助手。在安全的情况下，始终尽可能提供帮助。你的回答不应包括任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保你的回答是社会公正和积极的。
如果一个问题没有任何意义，或者事实上不连贯，请解释原因，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息，所有回答尽可能使用中文来回答。
"""
SYSTEM_EN = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


class Chat:
    '''聊天类
    :param checkpoint_path: str, 模型权重地址，可以是所在文件夹、文件地址、文件地址列表
    :param precision: bool, 精度
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quantization_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'max_length':2048, 'default_rtype':'logits', 'use_states':True}

    Examples:
    ```python
    >>> # 以chatglm2的命令行聊天
    >>> from bert4torch.pipelines import ChatGlm2Cli

    >>> checkpoint_path = "E:/pretrain_ckpt/glm/chatglm2-6b"
    >>> generation_config  = {'mode':'random_sample',
    ...                     'max_length':2048, 
    ...                     'default_rtype':'logits', 
    ...                     'use_states':True
    ...                     }
    >>> chat = ChatGlm2Cli(checkpoint_path, **generation_config)
    >>> chat.run()
    ```
    '''
    def __init__(self, checkpoint_path:str, precision:Literal['double', 'float', 'half', 'float16', 'bfloat16', None]=None, 
                 quantization_config:dict=None, generation_config:dict=None, create_model_at_startup:bool=True, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        self.config_path = kwargs.get('config_path', checkpoint_path)
        # generation_config顺序：config -> 显式传入generation_config -> kwargs
        config_path_tmp = get_config_path(self.config_path, allow_none=True)
        if config_path_tmp is not None:
            self.generation_config = json.load(open(config_path_tmp)).get('generation_config', dict())
        else:
            self.generation_config = dict()
        self.generation_config.update(generation_config if generation_config is not None else kwargs)
        self.precision = precision
        self.quantization_config = quantization_config
        self.tokenizer = self.build_tokenizer(**self.generation_config.get('tokenizer_config', dict()))
        self.generation_config['tokenizer'] = self.tokenizer
        if create_model_at_startup:
            self.model = self._build_model()
        self.build_other_config(**kwargs)

    def no_history_states(self) -> bool:
        '''不使用history的states'''
        return self.generation_config.get('states') is None
    
    def build_prompt(self, query:str, history:List[dict]) -> str:
        '''对query和history进行处理，生成进入模型的text
        :param query: str, 最近的一次user的input
        :param history: List, 历史对话记录，格式为[(input1, response1), (input2, response2)]
        '''
        raise NotImplementedError
    
    def build_tokenizer(self, **kwargs):
        '''初始化tokenizer'''
        from transformers import AutoTokenizer
        init_kwargs = {'additional_special_tokens'}
        new_kwargs = {k:v for k, v in kwargs.items() if k in init_kwargs}
        return AutoTokenizer.from_pretrained(self.config_path, trust_remote_code=True, **new_kwargs)

    def build_model(self):
        '''初始化model, 方便外部继承'''
        # 初始化
        model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path)
        model.eval()

        # 精度
        if self.precision == 'double':
            model = model.double()
        elif self.precision == 'float':
            model = model.float()
        elif self.precision in {'half', 'float16'}:
            model = model.half()
        elif self.precision == 'bfloat16':
            model = model.bfloat16()

        # 量化
        if self.quantization_config is not None:
            model = model.quantize(**self.quantization_config)
        return model.to(self.device)

    def _build_model(self):
        if (not hasattr(self, 'model')) or (self.model is None):
            self.model = self.build_model()

        elif self.device not in str(self.model.device):
            # 切换device到cuda上
            cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            log_info(f'{cur} - Moving model from cpu to {self.device}')
            self.model.to(self.device)
            gc.collect()
            cuda_empty_cache()
            
        return self.model
    
    def build_other_config(self, **kwargs):
        pass
    
    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        '''对response和histry进行后处理
        1. 可自行继承后来自定义
        2. history是本地修改的
        '''
        def process_history(res):
            if history is None:
                return
            elif len(history) == 0:
                raise ValueError('history len can not be 0')
            elif history[-1]['role'] == 'user':
                history.append({"role": "assistant", "content": res})
            elif history[-1]['role'] == 'assistant':
                history[-1]["content"] = res

        if isinstance(response, str):
            process_history(response)
            return response
        elif isinstance(response, (tuple, list)):  # response, states
            assert len(response) == 2
            self.generation_config['states'] = response[1]
            process_history(response[0])
            return response[0]
        else:
            raise TypeError('`response` type error')

    def chat(self, query:Union[str,list], history:List[dict]=None) -> str:
        '''chat模型使用, 配合对话模板使用'''
        history = history or []
        if isinstance(query, str):
            prompt = self.build_prompt(query, history)
        elif isinstance(query, list):
            prompt = [self.build_prompt(q, history) for q in query]
        self.model = self._build_model()
        response = self.model.generate(prompt, **self.generation_config)
        return self.process_response_history(response, history=history)

    def stream_chat(self, query:str, history:List[dict]=None):
        '''chat模型使用, 配合对话模板使用, 单条样本stream输出预测的结果'''
        history = history or []
        prompt = self.build_prompt(query, history)
        self.model = self._build_model()
        for response in self.model.stream_generate(prompt, **self.generation_config):
            yield self.process_response_history(response, history)

    def generate(self, query:str) -> str:
        '''base模型使用'''
        self.model = self._build_model()
        response = self.model.generate(query, **self.generation_config)
        return response

    def stream_generate(self, query:str):
        '''base模型使用, 单条样本stream输出预测的结果'''
        self.model = self._build_model()
        for response in self.model.stream_generate(query, **self.generation_config):
            yield response


class ChatCli(Chat):
    '''在命令行中交互的demo
    :param init_str: str, 对话问候句
    '''
    def build_other_config(self, **kwargs):
        self.init_str = kwargs.get('init_str', "输入内容进行对话，clear清空对话历史，stop终止程序")

    def build_cli_text(self, history:List[dict]) -> str:
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for query_or_response in history:
            # 现在的dict格式，形如{'role': 'user', 'content': '你好啊'}
            if query_or_response['role'] == "user":
                prompt += f"\n\nUser：{query_or_response['content']}"
            elif query_or_response['role'] == "assistant":
                # content_format主要用于content的结构化展示
                response = query_or_response.get('content_format', query_or_response['content'])
                prompt += f"\n\nAssistant：{response}"
        return prompt

    def run(self, stream:bool=True):
        import platform
        os_name = platform.system()
        history = []
        clear_command = 'cls' if os_name == 'Windows' else 'clear'
        print(self.init_str)
        while True:
            query = input("\nUser: ")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                history = []
                if 'states' in self.generation_config:
                    self.generation_config.pop('states')
                cuda_empty_cache()
                os.system(clear_command)
                print(self.init_str)
                continue
            
            prompt = self.build_prompt(query, history)
            self.model = self._build_model()
            # history是human和assistant的聊天历史
            # 格式如[('你好', '有什么可以帮您的？'), ('你是谁？', '我是一款人工智能助手。')]
            # 或者[{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '有什么可以帮您的？'}]
            if stream:
                for response in self.model.stream_generate(prompt, **self.generation_config):
                    response = self.process_response_history(response, history)
                    os.system(clear_command)
                    print(self.build_cli_text(history), flush=True)
            else:
                response = self.model.generate(prompt, **self.generation_config)
                response = self.process_response_history(response, history)
                os.system(clear_command)
                print(self.build_cli_text(history), flush=True)
            cuda_empty_cache()


class ChatWebGradio(Chat):
    '''gradio实现的网页交互的demo
    默认是stream输出，默认history不会删除，需手动清理
    '''    
    def build_other_config(self, max_length:int=4096, **kwargs):
        import gradio as gr
        self.gr = gr
        self.max_length = max_length
        self.max_repetition_penalty = 10
        self.stream = True  # 一般都是流式，因此未放在页面配置项
        if version.parse(gr.__version__) < version.parse("3.44.4"):
            log_warn_once('`gradio` changes frequently, the code is successfully tested under 3.44.4')

    def reset_user_input(self):
        return self.gr.update(value='')

    def reset_state(self):
        if 'states' in self.generation_config:
            self.generation_config.pop('states')
        cuda_empty_cache()  # 清理显存
        return [], []

    def set_generation_config(self, max_length:int, top_p:float, temperature:float, repetition_penalty:float):
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
        self.model = self._build_model()
        for response in self.model.stream_generate(input_text, **self.generation_config):
            response = self.process_response_history(response, history)
            chatbot[-1] = (input, str(response))
            yield chatbot, history
        cuda_empty_cache()  # 清理显存

    def __predict(self, input, chatbot, history, max_length, top_p, temperature, repetition_penalty):
        '''一次性生成'''
        self.set_generation_config(max_length, top_p, temperature, repetition_penalty)
        chatbot.append((input, ""))
        input_text = self.build_prompt(input, history)
        self.model = self._build_model()
        response = self.model.generate(input_text, **self.generation_config)
        response = self.process_response_history(response, history)
        chatbot[-1] = (input, response)
        cuda_empty_cache()  # 清理显存
        return chatbot, history

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


class ChatWebStreamlit(Chat):
    def build_other_config(self, max_length:int=4096, **kwargs):
        if not is_streamlit_available():
            raise ModuleNotFoundError('pip install streamlit')
        if version.parse(st.__version__) < version.parse("1.29.0"):
            log_warn_once('`streamlit` is successfully tested under 1.29.0')
        st.set_page_config(
            page_title="Chabot Web Demo",
            page_icon=":robot:",
            layout="wide"
        )
        self.max_length = max_length

    @st.cache_resource
    def _build_model(_self):
        return super()._build_model()
    
    @st.cache_resource
    def build_tokenizer(_self):
        return super().build_tokenizer()
    
    def run(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "states" not in st.session_state:
            st.session_state.states = None

        max_length = st.sidebar.slider("max_length", 0, self.max_length, self.max_length//2, step=1)
        top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

        buttonClean = st.sidebar.button("清理会话历史", key="clean")
        if buttonClean:
            st.session_state.history = []
            st.session_state.states = None
            cuda_empty_cache()
            st.rerun()

        for i, message in enumerate(st.session_state.history):
            with st.chat_message(name="user", avatar="user"):
                st.markdown(message[0])

            with st.chat_message(name="assistant", avatar="assistant"):
                st.markdown(message[1])

        with st.chat_message(name="user", avatar="user"):
            input_placeholder = st.empty()
        with st.chat_message(name="assistant", avatar="assistant"):
            message_placeholder = st.empty()

        prompt_text = st.chat_input("请输入您的问题")
        if prompt_text:
            input_placeholder.markdown(prompt_text)
            history = st.session_state.history
            states = st.session_state.states
            self.generation_config['max_length'] = max_length
            self.generation_config['top_p'] = top_p
            self.generation_config['temperature'] = temperature
            self.generation_config['states'] = states

            input_text = self.build_prompt(prompt_text, history)
            for response in self.model.stream_generate(input_text, **self.generation_config):
                response = self.process_response_history(response, history)
                message_placeholder.markdown(response)
            st.session_state.history = history + [(prompt_text, response)]
            st.session_state.states = self.generation_config.get('states')



# ==========================================================================================
# =========================              openai server          ============================
# ==========================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    cuda_empty_cache()


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: str
    content: str


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = 'default'
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


class ChatOpenaiApi(Chat):
    """OpenAi的api

    :param checkpoint_path: str, 模型所在的文件夹地址
    :param name: str, 模型名称
    :param generation_config: dict, 模型generate的参数设置
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    :param offload_when_nocall: str, 是否在一定时长内无调用就卸载模型，可以卸载到内存和disk两种
    :param max_callapi_interval: int, 最长调用间隔
    :param scheduler_interval: int, 定时任务的执行间隔
    :param api_keys: List[str], api keys的list

    Examples:
    ```python
    >>> # 以chatglm2的api部署为例
    >>> from bert4torch.pipelines import ChatGlm2OpenaiApi

    >>> checkpoint_path = "E:/pretrain_ckpt/glm/chatglm2-6b"
    >>> generation_config  = {'mode':'random_sample',
    ...                     'max_length':2048, 
    ...                     'default_rtype':'logits', 
    ...                     'use_states':True
    ...                     }
    >>> chat = ChatGlm2OpenaiApi(checkpoint_path, **generation_config)
    >>> chat.run()
    ```

    TODO:
    1. 在后续调用服务，模型从cpu转到cuda上时，内存不下降，猜测是因为不同线程中操作导致的
    2. 偶然会发生调用的时候，主线程和定时线程打架，导致device不一致的错误
    3. 如何offload到disk上，不占用内存和显存
    """
    def __init__(self, checkpoint_path:str, name:str='default', route_api:str='/chat/completions', route_models:str='/models', 
                 max_callapi_interval:int=24*3600, scheduler_interval:int=10*60, offload_when_nocall:Literal['cpu', 'disk']=None, 
                 api_keys:List[str]=None, **kwargs):
        self.offload_when_nocall = offload_when_nocall
        if offload_when_nocall is not None:
            kwargs['create_model_at_startup'] = False
        super().__init__(checkpoint_path, **kwargs)
        if not is_fastapi_available():
            raise ModuleNotFoundError("No module found, use `pip install fastapi`")
        from sse_starlette.sse import EventSourceResponse
        import sse_starlette
        if version.parse(sse_starlette.__version__) > version.parse('1.8'):
            log_warn('Module `sse_starlette` above 1.8 not support stream output')
        self.max_callapi_interval = max_callapi_interval  # 最长调用间隔
        self.scheduler_interval = scheduler_interval
        self.api_keys = api_keys
        self.EventSourceResponse = EventSourceResponse
        self.name = name
        self.role_user = 'user'
        self.role_assistant = 'assistant'

        if offload_when_nocall is None:
            self.app = FastAPI(lifespan=lifespan)
        else:
            # 启用后台任务，监控接口调用次数
            self.app = FastAPI()
            self.app.add_event_handler("startup", self.startup_event)
            self.app.add_event_handler("shutdown", lambda: self.shutdown_event(self.app.state.scheduler))
            self.lock = threading.Lock()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 添加路由
        router = APIRouter()
        router.add_api_route(route_models, methods=['GET'], endpoint=self.list_models, response_model=ModelList)
        router.add_api_route(route_api, methods=['POST'], endpoint=self.create_chat_completion, response_model=ChatCompletionResponse, 
                             dependencies=[Depends(self.check_api_key)])
        self.app.include_router(router)

    async def check_api_key(self, auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
        if self.api_keys is not None:
            if auth is None or (token := auth.credentials) not in self.api_keys:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "invalid_api_key",
                        }
                    },
                )
            return token
        else:
            # api_keys not set; allow all
            return None
    
    def startup_event(self):
        from apscheduler.schedulers.background import BackgroundScheduler  
        scheduler = BackgroundScheduler()  
        scheduler.add_job(self.check_last_call, 'interval', seconds=self.scheduler_interval)
        scheduler.start()
        self.app.state.scheduler = scheduler  # 将调度器存储在app的状态中，以便在shutdown时使用  
    
    def shutdown_event(scheduler):  
        if scheduler:  
            scheduler.shutdown()

    async def list_models(self):
        model_card = ModelCard(id=self.name)
        return ModelList(data=[model_card])

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.model != self.name:
            raise HTTPException(status_code=404, detail="Invalid model")

        if request.temperature:
            self.generation_config['temperature'] = request.temperature
        if request.top_p:
            self.generation_config['top_p'] = request.top_p
        if request.top_k:
            self.generation_config['top_k'] = request.top_k
        if request.max_length:
            self.generation_config['max_length'] = request.max_length
        if request.repetition_penalty:
            self.generation_config['repetition_penalty'] = request.repetition_penalty

        if request.messages[-1].role != self.role_user:  # 最后一条msg的role必须是user
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content
        history = request.messages[:-1]
        input_text = self.build_prompt(query, history)
        
        if self.offload_when_nocall is None:
            self.model = self._build_model()
        else:
            with self.lock:
                self.model = self._build_model()
            self.last_callapi_timestamp = time.time()

        # 流式输出
        if request.stream:
            generate = self.predict(input_text, request.model, history)
            return self.EventSourceResponse(generate, media_type="text/event-stream")
        
        # 非流式输出
        else:
            response = self.model.generate(input_text, **self.generation_config)
            response = self.process_response_history(response, history)
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role=self.role_assistant, content=response),
                finish_reason="stop"
            )

            return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, query: str, model_id: str, history:list):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=self.role_assistant),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        current_length = 0

        for new_response in self.model.stream_generate(query, **self.generation_config):
            if len(new_response) == current_length:
                continue

            new_text = self.process_response_history(new_response[current_length:], history)
            current_length = len(new_response)

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))


        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop"
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        yield '[DONE]'

    def check_last_call(self):
        '''检测距离上一次调用超出规定时间段'''
        now = time.time()
        if not hasattr(self, 'model')  or (self.model is None):
            return
        elif not hasattr(self, 'last_callapi_timestamp'):
            self.last_callapi_timestamp = now
        elif now - self.last_callapi_timestamp > self.max_callapi_interval:  # 超出调用间隔
            if (self.offload_when_nocall == 'cpu') and (str(self.model.device) != 'cpu'):
                with self.lock:
                    # 如果没有调用，将模型转移到CPU
                    self.model.to('cpu')
                    cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    log_info(f"{cur} - Model moved to cpu due to no activity for {self.max_callapi_interval} sec.")
                    gc.collect()
                    cuda_empty_cache()
            elif (self.offload_when_nocall == 'disk') and hasattr(self, 'model'):
                with self.lock:
                    self.model = None
                    del self.model
                    cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    log_info(f"{cur} - Model moved to disk due to no activity for {self.max_callapi_interval} sec.")
                    gc.collect()
                    cuda_empty_cache()

    def run(self, app:str=None, host:str="0.0.0.0", port:int=8000, **kwargs):
        '''主程序入口'''
        import uvicorn
        uvicorn.run(app or self.app, host=host, port=port, **kwargs)


class ChatOpenaiClient:
    '''使用openai来调用
    
    Examples:
    ```python
    >>> messages = [
    ...         {"content": "你好", "role": "user"},
    ...         {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
    ...         {"content": "你可以做什么？", "role": "user"}
    ...         ]
    >>> client = ChatOpenaiClient('http://127.0.0.1:8000')

    >>> # 流式
    >>> for token in client.stream_chat(messages):
    ...     print(token, end='', flush=True)

    >>> # 非流式
    >>> print(client.chat(messages))
    ```
    '''
    def __init__(self, base_url:str, api_key:str=None, **kwargs) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, **kwargs)
    
    def stream_chat(self, messages:List[Dict], model:str='default', max_length:int=None, temperature:float=None, top_p:float=None, **kwargs):
        '''流式返回'''
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
            )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    def chat(self, messages:List[Dict], model:str='default', max_length:int=None, temperature:float=None, top_p:float=None, **kwargs):
        '''一次性返回'''
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
            )
        content = response.choices[0].message.content
        return content
    
    def stream_chat_cli(self, *args, **kwargs):
        for token in self.stream_chat(*args, **kwargs):
            print(token, end='', flush=True)


class ChatOpenaiClientSseclient:
    '''调用openai接口的client, 流式请求

    Examples:
    ```python
    >>> # 注意事项：部分调用时候有额外参数传入，如下：
    >>> client = ChatOpenaiClientSseclient(url='https://chatpet.openai.azure.com/openai/deployments/chatGPT-turbo16K/chat/completions', 
    ...                                 header={'api-key': "填写对应的api-key"},
    ...                                 params={'api-version': '2023-03-15-preview'})

    >>> body = {
    ...         "messages": [
    ...             {"content": "你好", "role": "user"},
    ...             {"content": "你好，我是法律大模型", "role": "assistant"},
    ...             {"content": "基金从业可以购买股票吗", "role": "user"}],
    ...         "model": "default",
    ...         "stream": True
    ...     }

    >>> client = ChatOpenaiClientSseclient('http://127.0.0.1:8000')
    >>> # 测试打印
    >>> client.stream_chat_cli(body)

    >>> # 流式
    >>> for token in client.stream_chat(body):
    ...     print(token, end='', flush=True)
    ```
    '''
    def __init__(self, url:str, api_key:str=None, header:dict=None, params:dict=None) -> None:
        self.url = url
        self.api_key = api_key
        self.header = header
        self.params = params

        if is_sseclient_available():
            import sseclient
        else:
            raise ModuleNotFoundError('No module found, you may `pip install sseclient-py`')
        
        self.sseclient = sseclient
   
    def stream_chat(self, body, **kwargs):
        '''接口调用'''
        reqHeaders = {'Accept': 'text/event-stream'}
        if self.api_key is not None:
            reqHeaders["Authorization"] = f"Bearer {self.api_key}"

        if self.header is not None:
            reqHeaders.update(self.header)
        
        request = requests.post(self.url, stream=True, headers=reqHeaders, json=body, params=self.params, **kwargs)
        client = self.sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    yield data['content']

    def stream_chat_cli(self, body, **kwargs):
        '''简单测试在命令行打印'''
        for token in self.stream_chat(body, **kwargs):
            print(token, end='', flush=True)


# ==========================================================================================
# =========================              各个具体模型实现        ============================
# ==========================================================================================

class ChatGlm(Chat):
    def build_prompt(self, query:str, history:List[dict]) -> str:
        # 没有system和function call
        if not history:
            prompt = query
        else:
            prompt, turn_i = "", 0
            if self.no_history_states():
                for query_or_response in enumerate(history):
                    if query_or_response['role'] == 'user':
                        prompt += f"[Round {turn_i}]\n问：{query_or_response['content']}\n"
                    elif query_or_response['role'] == 'assistant':
                        prompt += f"答：{query_or_response['content']}\n"
                        turn_i += 1
            else:
                prompt += self.generation_config['states']['last_token']

            prompt += f"[Round {turn_i}]\n问：{query}\n答："
        history.append({'role': 'user', 'content': query})
        return prompt
    
    def process_response_history(self, response, history):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        response = super().process_response_history(response, history)
        return response

class ChatGlmCli(ChatGlm, ChatCli): pass
class ChatGlmWebGradio(ChatGlm, ChatWebGradio): pass
class ChatGlmWebStreamlit(ChatGlm, ChatWebStreamlit): pass
class ChatGlmOpenaiApi(ChatGlm, ChatOpenaiApi): pass


class ChatGlm2(Chat):
    def build_prompt(self, query, history:List[dict]):
        # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
        prompt, turn_i = "", 1
        if self.no_history_states():
            for query_or_response in enumerate(history):
                if query_or_response['role'] == 'user':
                    prompt += f"[Round {turn_i}]\n\n问：{query_or_response['content']}\n\n"
                elif query_or_response['role'] == 'assistant':
                    prompt += f"答：{query_or_response['content']}\n"
                    turn_i += 1
        else:
            prompt += self.generation_config['states']['last_token']

        prompt += f"[Round {turn_i}]\n\n问：{query}\n\n答："
        history.append({'role': 'user', 'content': query})
        return prompt
    
    def process_response_history(self, response:str, history:List[dict]):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        response = super().process_response_history(response, history)
        return response

class ChatGlm2Cli(ChatGlm2, ChatCli): pass
class ChatGlm2WebGradio(ChatGlm2, ChatWebGradio): pass
class ChatGlm2WebStreamlit(ChatGlm2, ChatWebStreamlit): pass
class ChatGlm2OpenaiApi(ChatGlm2, ChatOpenaiApi): pass


class ChatGlm3(Chat):
    def build_other_config(self, system:str=None, tools:dict=None, **kwargs):
        super().build_other_config(**kwargs)
        self.system = system
        self.tools = tools

    def build_prompt(self, query:str, history:List[dict]):       
        if (len(history) == 0) or (history[0]["role"] != "system"):
            # 增加system信息
            if self.system is not None:
                history.insert(0, {"role": "system", "content": self.system})
            # 增加tools信息
            if self.tools is not None and self.system is not None:
                history[0]['tools'] = self.tools
            elif self.tools is not None and self.system is None:
                history.insert(0, {
                    "role": "system", 
                    "content": "Answer the following questions as best as you can. You have access to the following tools:",
                    "tools": self.tools
                })

        if self.no_history_states():
            # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
            input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
        else:
            input_ids += self.generation_config['states']['last_token']
        history.append({"role": "user", "content": query})
        return input_ids
        
    def process_response_history(self, response:str, history:list):
        response = super().process_response_history(response, history)
        if (not response) or (response[-1] == "�"):
            return response

        content = ""
        for resp in response.split("<|assistant|>"):
            try:
                # 使用tools时候，stream_generate会有问题，因为中间结果是无法结构化解析的
                metadata, content = resp.split("\n", maxsplit=1)
                if not metadata.strip():
                    content = content.strip()
                    history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                    content = content.replace("[[训练时间]]", "2023年")
                else:
                    history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                    if history[0]["role"] == "system" and "tools" in history[0]:
                        content = "\n".join(content.split("\n")[1:-1])
                        def tool_call(**kwargs):
                            return kwargs
                        parameters = eval(content)
                        content = {"name": metadata.strip(), "parameters": parameters}
                    else:
                        content = {"name": metadata.strip(), "content": content}
                    history[-1]['content_format'] = content
            except (ValueError, SyntaxError):  # 同事对应split和eval两种错误
                content = resp.strip()
                history[-1] = {"role": "assistant", "metadata": "", "content": content}
        return content

class ChatGlm3Cli(ChatGlm3, ChatCli): pass
class ChatGlm3WebGradio(ChatGlm3, ChatWebGradio): pass
class ChatGlm3WebStreamlit(ChatGlm3, ChatWebStreamlit): pass
class ChatGlm3OpenaiApi(ChatGlm3, ChatOpenaiApi): pass


class ChatGlm4(Chat):
    def build_other_config(self, system:str=None, tools:dict=None, **kwargs):
        super().build_other_config(**kwargs)
        self.system = system
        self.tools = tools

    def build_prompt(self, query, history:list):
        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        history.append({"role": "user", "content": query})
        if self.no_history_states():
            input_ids = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True,
                                                           return_tensors="pt", return_dict=True)['input_ids']
        else:
            input_ids += self.generation_config['states']['last_token']
        return input_ids
    
    def process_response_history(self, response:str, history:list):
        response = super().process_response_history(response, history)
        if (not response) or (response[-1] == "�"):
            return response

        content = ""
        for resp in response.split("<|assistant|>"):
            if "\n" in resp:
                metadata, content = resp.split("\n", maxsplit=1)
            else:
                metadata, content = "", resp
            if not metadata.strip():
                content = content.strip()
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                content = content.replace("[[训练时间]]", "2024年")
            else:
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                if history[0]["role"] == "system" and "tools" in history[0]:
                    try:
                        parameters = json.loads(content)
                        content = {"name": metadata.strip(), "parameters": parameters}
                    except json.JSONDecodeError:
                        content = {"name": metadata.strip(), "content": content}
                else:
                    content = {"name": metadata.strip(), "content": content}
                history[-1]['content_format'] = content
        return content

class ChatGlm4Cli(ChatGlm4, ChatCli): pass
class ChatGlm4WebGradio(ChatGlm4, ChatWebGradio): pass
class ChatGlm4WebStreamlit(ChatGlm4, ChatWebStreamlit): pass
class ChatGlm4OpenaiApi(ChatGlm4, ChatOpenaiApi): pass


class ChatInternLM(Chat):
    def build_prompt(self, query, history:list):
        prompt = ""
        if self.no_history_states():
            for user, bot in history:
                prompt += f"""<s><|User|>:{user}<eoh>\n<|Bot|>:{bot}<eoa>\n"""
        else:
            prompt += self.generation_config['states']['last_token']

        if len(prompt) == 0:
            prompt += "<s>"
        if query is not None:
            prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return prompt

    def process_response_history(self, response, history=None):
        response = super().process_response_history(response, history)
        for reg in ['<s>', '</s>', '<eoh>', '<eoa>']:
            response = response.replace(reg, '')
        return response

class ChatInternLMCli(ChatInternLM, ChatCli): pass
class ChatInternLMWebGradio(ChatInternLM, ChatWebGradio): pass
class ChatInternLMWebStreamlit(ChatInternLM, ChatWebStreamlit): pass
class ChatInternLMOpenaiApi(ChatInternLM, ChatOpenaiApi): pass


class ChatQwen(Chat):
    def build_other_config(self, system:str=None, max_window_size=6144, **kwargs):
        super().build_other_config(**kwargs)
        self.system = system if system is not None else SYSTEM_ZH
        self.max_window_size = max_window_size

    def build_prompt(self, query:str, history:List[dict]) -> str:
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", self.system)
        raw_text = ""

        if self.no_history_states():
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
        else:
            raw_text += self.generation_config['states']['last_token']

        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text

class ChatQwenCli(ChatQwen, ChatCli): pass
class ChatQwenWebGradio(ChatQwen, ChatWebGradio): pass
class ChatQwenWebStreamlit(ChatQwen, ChatWebStreamlit): pass
class ChatQwenOpenaiApi(ChatQwen, ChatOpenaiApi): pass


class ChatLLaMA2(Chat):
    def build_other_config(self, system:str=None, **kwargs):
        super().build_other_config(**kwargs)
        self.system = system if system is not None else SYSTEM_EN

    def build_prompt(self, query:str, history:List[dict]) -> str:
        if self.no_history_states():
            texts = [f'[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n']
            for user_input, response in history:
                texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        else:
            texts = [self.generation_config['states']['last_token']]

        texts.append(f'{query.strip()} [/INST]')
        return ''.join(texts)

class ChatLLaMA2Cli(ChatLLaMA2, ChatCli): pass
class ChatLLaMA2WebGradio(ChatLLaMA2, ChatWebGradio): pass
class ChatLLaMA2WebStreamlit(ChatLLaMA2, ChatWebStreamlit): pass
class ChatLLaMA2OpenaiApi(ChatLLaMA2, ChatOpenaiApi): pass


class ChatLLaMA3(Chat):
    def build_other_config(self, system:str=None, **kwargs):
        super().build_other_config(**kwargs)
        self.system = system if system is not None else SYSTEM_ZH

    def build_prompt(self, query:str, history:List[dict]) -> str:
        if self.no_history_states():
            messages = [{"role": "system", "content": self.system}]
            for user_input, response in history:
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": query})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            texts = self.generation_config['states']['last_token']
            texts += f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

class ChatLLaMA3Cli(ChatLLaMA3, ChatCli): pass
class ChatLLaMA3WebGradio(ChatLLaMA3, ChatWebGradio): pass
class ChatLLaMA3WebStreamlit(ChatLLaMA3, ChatWebStreamlit): pass
class ChatLLaMA3OpenaiApi(ChatLLaMA3, ChatOpenaiApi): pass


class ChatZiya(Chat):
    def build_prompt(self, query:str, history:List[dict]) -> str:
        prompt = ''
        if self.no_history_states():
            for human, bot in history:
                prompt += f"<human>:{human}\n<bot>:{bot}\n"
        else:
            prompt += self.generation_config['states']['last_token']
        
        prompt += f"<human>:{query.strip()}\n<bot>:"
        return prompt

class ChatZiyaCli(ChatZiya, ChatCli): pass
class ChatZiyaWebGradio(ChatZiya, ChatWebGradio): pass
class ChatZiyaWebStreamlit(ChatZiya, ChatWebStreamlit): pass
class ChatZiyaOpenaiApi(ChatZiya, ChatOpenaiApi): pass


class ChatChineseAlphaLLaMA(Chat):
    def build_other_config(self, system:str=None, **kwargs):
        super().build_other_config(**kwargs)
        if system is None:
            self.system = \
("Below is an instruction that describes a task. "
"Write a response that appropriately completes the request.\n\n"
)
        else:
            self.system = system

    def build_prompt(self, query:str, history:List[dict]) -> str:
        prompt = ''
        if self.no_history_states():
            for inst, resp in history:
                prompt += f"### Instruction:\n\n{inst}\n\n### Response:\n\n{resp}\n\n"
            prompt += f"### Instruction:\n\n{query}\n\n### Response:\n\n"
            prompt = self.system + prompt
        else:
            prompt += self.generation_config['states']['last_token'] + f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        return prompt

class ChatChineseAlphaLLaMACli(ChatChineseAlphaLLaMA, ChatCli): pass
class ChatChineseAlphaLLaMAWebGradio(ChatChineseAlphaLLaMA, ChatWebGradio): pass
class ChatChineseAlphaLLaMAWebStreamlit(ChatChineseAlphaLLaMA, ChatWebStreamlit): pass
class ChatChineseAlphaLLaMAOpenaiApi(ChatChineseAlphaLLaMA, ChatOpenaiApi): pass


class ChatBelle(Chat):
    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.checkpoint_path, use_fast=False)
    
    def build_prompt(self, query:str, history:List[dict]) -> str:
        prompt = ''
        if self.no_history_states():
            for item in history:
                prompt += f"Human: {item[0]} \n\nAssistant: {item[1]}\n\n"
        else:
            prompt += self.generation_config['states']['last_token']
        prompt += f"Human: {query} \n\nAssistant: "
        return prompt

class ChatBelleCli(ChatBelle, ChatCli): pass
class ChatBelleWebGradio(ChatBelle, ChatWebGradio): pass
class ChatBelleWebStreamlit(ChatBelle, ChatWebStreamlit): pass
class ChatBelleOpenaiApi(ChatBelle, ChatOpenaiApi): pass


class ChatBaichuan(Chat):
    def build_other_config(self, **kwargs):
        super().build_other_config(**kwargs)
        self.user_token_id = kwargs.get('user_token_id', 195)
        self.assistant_token_id = kwargs.get('assistant_token_id', 196)

    def build_prompt(self, query:str, history:List[dict]) -> str:
        total_input = []
        if self.no_history_states():
            for user, assistant in history:
                total_input += [self.user_token_id] + self.tokenizer.encode(user)  
                total_input += [self.assistant_token_id] + self.tokenizer.encode(assistant) + [self.tokenizer.eos_token_id]
        else:
            total_input += [self.generation_config['states']['last_token_id']]
        total_input += [self.user_token_id] + self.tokenizer.encode(query)
        total_input.append(self.assistant_token_id)
        return total_input

class ChatBaichuanCli(ChatBaichuan, ChatCli): pass
class ChatBaichuanWebGradio(ChatBaichuan, ChatWebGradio): pass
class ChatBaichuanWebStreamlit(ChatBaichuan, ChatWebStreamlit): pass
class ChatBaichuanOpenaiApi(ChatBaichuan, ChatOpenaiApi): pass
