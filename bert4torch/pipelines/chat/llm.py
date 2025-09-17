''' 大模型聊天的pipeline调用
主要功能：
1. 命令行调用各个模型demo
2. 利用fastapi为大模型搭建openai格式的server和client调用
    Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
    Usage: python openai_api.py
    Visit http://localhost:8000/docs for documents.
3. web界面快速搭建demo(gradio+streamlit)

# TODO: 设置return_states=True时候，受到build_prompt影响，很难保证prompt完全复现
这里采用添加self.generation_config['states']['last_token']，是因为推理完成可能是因为到达max_length，未必是遇到了eos
'''

import os
import torch
from typing import Union, Optional, List, Tuple, Literal, Dict
from bert4torch.pipelines.base import PipeLineBase
from bert4torch.snippets import (
    log_warn_once, 
    get_config_path, 
    log_free,
    log_info, 
    log_info_once,
    log_warn, 
    log_error,
    colorful,
    cuda_empty_cache,
    is_fastapi_available, 
    is_pydantic_available, 
    is_streamlit_available,
    is_package_available,
    has_chinese_char,
    add_start_docstrings,
    JsonConfig,
    NoopContextManager,
    sequence_padding,
    DottableDict
)
from packaging import version
import gc
import time
import json
from contextlib import asynccontextmanager
import threading
import re
import copy
from .conversation import Conversation


class NoneObject:
    def __init__(self, *args, **kwarg):
        pass


if is_fastapi_available():
    from fastapi import FastAPI, HTTPException, APIRouter, Depends
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.middleware.cors import CORSMiddleware
else:
    class FastAPI: pass
    class HTTPAuthorizationCredentials: pass
    Depends, HTTPBearer = NoneObject, NoneObject

if is_pydantic_available():
    from pydantic import BaseModel, Field
else:
    BaseModel, Field = object, NoneObject


if is_streamlit_available():
    import streamlit as st
else:
    # 防止streamlit不存在时候报错
    import bert4torch.snippets as st
    st.cache_resource = st.delete_arguments


__all__ = [
    'Chat',
    'Glm',
    'Glm2',
    'Glm3',
    'Glm4',
    'InternLM',
    'InternLM2',
    'Qwen',
    'Qwen2',
    'LLaMA2',
    'LLaMA3',
    'ApplyChatTemplate',
    'Ziya',
    'ChineseLlamaAlpaca',
    'Belle',
    'Baichuan',
    'PretrainedTextContinuation'
    ]


# 一些通用的system话术
SYSTEM_ZH = """你是一个乐于助人、尊重他人、诚实的中文聊天助手。在安全的情况下，始终尽可能提供帮助。你的回答不应包括任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保你的回答是社会公正和积极的。
如果一个问题没有任何意义，或者事实上不连贯，请解释原因，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息，所有回答尽可能使用中文来回答。
"""
SYSTEM_EN = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

CHAT_START_DOCSTRING = r"""
    :param checkpoint_path: str, 模型权重地址，可以是所在文件夹、文件地址、文件地址列表
    :param torch_dtype: bool, 精度, 'double', 'float', 'half', 'float16', 'bfloat16'
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quant_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'max_length':2048, 'default_rtype':'logits', 'use_states':True}
    :param create_model_at_startup: bool, 是否在启动的时候加载模型, 默认为True
    :param system: Optional[str]=None, 模型使用的system信息, 仅部分模型可用, 且openai api格式的不需要设置该参数
"""

# ==========================================================================================
# =========================                 基类                ============================
# ==========================================================================================
@add_start_docstrings(CHAT_START_DOCSTRING)
class ChatBase(PipeLineBase):
    def __init__(self, checkpoint_path:str, config_path:str=None, 
                 torch_dtype:Literal['double', 'float', 'half', 'float16', 'bfloat16', None]=None, 
                 quantization_config:dict=None, generation_config:dict=None, 
                 create_model_at_startup:bool=True, system:str=None, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or checkpoint_path
        # generation_config顺序：config -> 显式传入generation_config -> kwargs
        config_path_tmp = get_config_path(self.config_path, allow_none=True)
        if config_path_tmp is not None:
            self.config = JsonConfig(config_path_tmp)
            self.generation_config = self.config.get('generation_config', dict())
        else:
            self.config = DottableDict()
            self.generation_config = dict()
        self.generation_config.update(generation_config if generation_config is not None else kwargs)
        self.torch_dtype = torch_dtype
        self.quantization_config = quantization_config
        kwargs['return_dict'] = False
        if create_model_at_startup:
            self.model = self.build_model(**kwargs)
        # tokenizer放在build_model之后，防止用户传入的是模型名称需要下载
        self.tokenizer = self.build_tokenizer(**self.generation_config.get('tokenizer_config', dict()))
        self.generation_config['tokenizer'] = self.tokenizer
        self.template_config = self.config.get('template_config', dict())
        self.system = system
        self.kwargs = kwargs

    def no_history_states(self) -> bool:
        '''不使用history的states'''
        return self.generation_config.get('states') is None
    
    def build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    def build_template(self, **kwargs):
        kwargs = {k:v for k,v in kwargs.items() if v}
        if self.template_config or kwargs:
            return Conversation(**{**self.template_config, **kwargs}).copy()
        return None
    
    def build_tokenizer(self, **kwargs):
        '''初始化tokenizer'''
        from transformers import AutoTokenizer
        init_kwargs = {'additional_special_tokens'}
        new_kwargs = {k:v for k, v in kwargs.items() if k in init_kwargs}
        try:
            return AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True, **new_kwargs)
        except Exception as e:
            _, transformer_version = is_package_available('transformers', return_version=True)
            config_tmp = os.path.join(self.checkpoint_path, 'config.json')
            request_version = JsonConfig(config_tmp).get('transformers_version') if os.path.exists(config_tmp) else None
            if request_version is not None:
                log_error(f'Please check your transformer=={transformer_version}, while transformer=={request_version} requested.')
            else:
                log_error(f'Please check your transformer=={transformer_version}, which may not compatible.')
            raise e
        
    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        '''对response和histry进行后处理
        1. 可自行继承后来自定义
        2. history是本地修改的, 用于命令行或者web demo下一次构建历史使用的, response可以不等于history[-1]['content']

        :param response: 大模型直接输出的字符串
        :param history: 聊天记录
            - role: 角色 
            - raw_content: 模型直接输出的结果，可用于cli或web demo的展示
            - content: 处理后的，多轮对话中用于prompt搭建
            - function_call: function调用

        Returns: 接口直接返回的值（处理后的response, 而不是模型直接输出的结果）
        '''
        def process_history(res:str):
            res = res.strip()
            if history is None:
                return
            elif len(history) == 0:
                raise ValueError('history len can not be 0')
            elif history[-1]['role'] == 'user':
                history.append({"role": "assistant", "content": res, "raw_content": res})
            elif history[-1]['role'] == 'assistant':
                history[-1]["content"] = res
                history[-1]["raw_content"] = res

        if isinstance(response, str):
            process_history(response)
            return response
        elif isinstance(response, (tuple, list)):  # response, states
            assert len(response) == 2
            self.generation_config['states'] = response[1]
            process_history(response[0])
            return response[0]
        else:
            raise TypeError(f'`response` type={type(response)} which is not supported')

    @staticmethod
    def update_history(history:list, query:str):
        history.append({"role": "user", "content": query})
        return history
    
    def chat(self, query:Union[str, List[str]], history:List[dict]=None, functions:List[dict]=None, 
             return_history:bool=False, **kwargs) -> Union[str, List[str]]:
        '''chat模型使用, 配合对话模板使用'''
        history = history or []

        if isinstance(query, str):
            # 单条输入
            prompts:Union[str, torch.Tensor] = self.build_prompt(query, history, functions)
            response = self.model.generate(prompts, **self.generation_config)
            if isinstance(response, str):
                # 生成单条输出
                response = self.process_response_history(response, history=history)
            elif isinstance(response, list):
                # 为单条query生成多条response
                response = [self.process_response_history(resp, history=copy.deepcopy(history)) for resp in response]
            else:
                raise TypeError(f'`response` type={type(response)} which is not supported')
            
        elif isinstance(query, list):
            # 多条输入
            history_cp = [copy.deepcopy(history) for _ in query]
            prompts:List[str] = [self.build_prompt(q, h, functions) for q, h in zip(query, history_cp)]
            if all([isinstance(i, torch.Tensor) for i in prompts]):
                # build_prompt返回的都是tokenize后的input_ids，需要concat+padding在一起
                prompts = sequence_padding(prompts, value=self.tokenizer.pad_token_id, padding_side='left')
            response = self.model.generate(prompts, **self.generation_config)
            response = [self.process_response_history(r, history=h) for r, h in zip(response, history_cp)]
        else:
            raise TypeError(f'Args `query` type={type(query)} which is not supported')
        if return_history:
            return response, history
        else:
            return response
        
    def stream_chat(self, query:str, history:List[dict]=None, functions:List[dict]=None, **kwargs):
        '''chat模型使用, 配合对话模板使用, 单条样本stream输出预测的结果'''
        history = history or []
        prompt = self.build_prompt(query, history, functions)
        for response in self.model.stream_generate(prompt, **self.generation_config):
            yield self.process_response_history(response, history)

    def generate(self, query:Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        '''base模型使用'''
        return self.model.generate(query, **self.generation_config)

    def stream_generate(self, query:str):
        '''base模型使用, 单条样本stream输出预测的结果'''
        yield from self.model.stream_generate(query, **self.generation_config)


# ==========================================================================================
# =========================              命令行聊天的基类        ============================
# ==========================================================================================
@add_start_docstrings(CHAT_START_DOCSTRING)
class ChatCli(ChatBase):
    '''在命令行中交互的demo
    :param init_str: str, 对话问候句
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_str = kwargs.get('init_str', "输入内容进行对话，clear清空对话历史，stop终止程序")

    def build_cli_text(self, history:List[dict]) -> str:
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for query_or_response in history:
            # 现在的dict格式，形如{'role': 'user', 'content': '你好啊'}
            if query_or_response['role'] == "user":
                prompt += f"\n\n{colorful('User：', color='green')}{query_or_response['content']}"
            elif query_or_response['role'] == "assistant":
                response = query_or_response.get('raw_content', query_or_response['content'])
                prompt += f"\n\n{colorful('Assistant：', color='red')}{response}"
                # function_call主要用于content的结构化展示
                if query_or_response.get('function_call'):
                    prompt += f"\n\n{colorful('Function：', color='yellow')}{query_or_response['function_call']}"
        return prompt

    def run(self, functions:List[dict]=None, stream:bool=True):
        import platform
        os_name = platform.system()
        history = []
        clear_command = 'cls' if os_name == 'Windows' else 'clear'
        print(self.init_str)
        while True:
            query = input(f"\n{colorful('User：', color='green')}")
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
            
            prompt = self.build_prompt(query, history, functions)
            # history是human和assistant的聊天历史
            # 格式如[{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '有什么可以帮您的？'}]
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


# ==========================================================================================
# =======================              网页Gradio聊天的基类        ==========================
# ==========================================================================================
@add_start_docstrings(CHAT_START_DOCSTRING)
class ChatWebGradio(ChatBase):
    '''gradio实现的网页交互的demo
    1. system和functions均在网页上进行设置(若有)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import gradio as gr
        self.gr = gr
        self.max_length = self.generation_config.get('max_length', 4096)
        self.max_repetition_penalty = kwargs.get('max_repetition_penalty', 10.0)
        self.max_temperature = kwargs.get('max_temperature', 10.0)
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

    def _stream_predict(self, query, chatbot, history, max_length, top_p, temperature, repetition_penalty, system, functions):
        '''流式生成'''
        self.set_generation_config(max_length, top_p, temperature, repetition_penalty)
        chatbot.append((query, ""))
        functions = self._set_system_functions(system, functions)
        input_text = self.build_prompt(query, history, functions)
        for response in self.model.stream_generate(input_text, **self.generation_config):
            response = self.process_response_history(response, history)
            if history[-1].get('raw_content'):
                response = history[-1]['raw_content']
            if history[-1].get('function_call'):
                response += f"\n\nFunction：{history[-1]['function_call']}"
            chatbot[-1] = (query, response)
            yield chatbot, history
        cuda_empty_cache()  # 清理显存

    def _set_system_functions(self, system:str=None, functions:List[dict]=None):
        '''设置system和functions参数'''
        try:
            if functions is not None and functions.strip() != '':
                functions = json.loads(functions)
            else:
                functions = None
        except json.JSONDecodeError:
            functions = None
            log_warn('Functions implement not json format')
        if system.strip() != '':
            self.system = system
        return functions

    def regenerate(self, query, chatbot, history, max_length, top_p, temperature, repetition_penalty, system, functions):
        if chatbot and history and history[-1]['role'] == 'assistant':
            query, _ = chatbot.pop()
            history.pop()
            history.pop()
            yield from self._stream_predict(query, chatbot, history, max_length, top_p, temperature, repetition_penalty, system, functions)
        else:
            return chatbot, history
    
    def run(self, host:str=None, port:int=None, **launch_configs):
        with self.gr.Blocks() as demo:
            self.gr.HTML("""<h1 align="center">Chabot Gradio Demo</h1>""")

            with self.gr.Row():
                with self.gr.Column(scale=4):
                    chatbot = self.gr.Chatbot()
                    with self.gr.Column(scale=12):
                        query = self.gr.Textbox(show_label=False, placeholder="Input...", lines=10, max_lines=10) # .style(container=False)
                    with self.gr.Row():
                        submitBtn = self.gr.Button("🚀 Submit", variant="primary")
                        regen_btn = self.gr.Button('🤔️ Regenerate')
                        emptyBtn = self.gr.Button("🧹 Clear History")

                with self.gr.Column(scale=1):
                    max_length = self.gr.Slider(0, self.max_length, value=self.max_length, step=1.0, label="max_length", interactive=True)
                    top_p = self.gr.Slider(0, 1, value=self.generation_config.get('top_p', 1.0), step=0.01, label="top_p", interactive=True)
                    temperature = self.gr.Slider(0, self.max_temperature, value=self.generation_config.get('temperature', 1.0), step=0.1, label="temperature", interactive=True)
                    repetition_penalty = self.gr.Slider(0, self.max_repetition_penalty, value=self.generation_config.get('repetition_penalty', 1.0), step=0.1, label="repetition_penalty", interactive=True)
                    system = self.gr.Textbox(label='System Prompt (If exists)', lines=6, max_lines=6)
                    functions = self.gr.Textbox(label='Functions Json Format (If exists)', lines=6, max_lines=6)

            history = self.gr.State([])
            _input_tuple = [query, chatbot, history, max_length, top_p, temperature, repetition_penalty, system, functions]
            submitBtn.click(self._stream_predict, _input_tuple, [chatbot, history], show_progress=True)
            submitBtn.click(self.reset_user_input, [], [query])
            regen_btn.click(self.regenerate, _input_tuple, [chatbot, history], show_progress=True)
            emptyBtn.click(self.reset_state, outputs=[chatbot, history], show_progress=True)

        demo.queue().launch(server_name = launch_configs.pop('server_name', host), 
                            server_port = launch_configs.pop('server_port', port), 
                            **launch_configs)


# ==========================================================================================
# =====================              网页streamlit聊天的基类        =========================
# ==========================================================================================
@add_start_docstrings(CHAT_START_DOCSTRING)
class ChatWebStreamlit(ChatBase):
    '''
    1. 启动方式: streamlit run app.py --server.address 0.0.0.0 --server.port 8001
    2. system和functions均在网页上进行设置(若有)
    '''
    def __init__(self, *args, **kwargs):
        if not is_streamlit_available():
            raise ModuleNotFoundError('pip install streamlit')
        if version.parse(st.__version__) < version.parse("1.29.0"):
            log_warn_once('`streamlit` is successfully tested under 1.29.0')
        st.set_page_config(
            page_title="Chabot Web Demo",
            page_icon=":robot:",
            layout="wide"
        )
        super().__init__(*args, **kwargs)
        log_warn_once('You should use command `streamlit run app.py --server.address 0.0.0.0 --server.port 8001` to launch')
        self.max_length = self.generation_config.get('max_length', 4096)
        self.max_repetition_penalty = kwargs.get('max_repetition_penalty', 10.0)
        self.max_temperature = kwargs.get('max_temperature', 10.0)

    @st.cache_resource
    def build_model(self, **kwarg):
        return super().build_model(**kwarg)
    
    @st.cache_resource
    def build_tokenizer(_self, **kwarg):
        return super().build_tokenizer(**kwarg)
    
    def run(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "states" not in st.session_state:
            st.session_state.states = None

        max_length = st.sidebar.slider("max_length", 0, self.max_length, self.max_length//2, step=1)
        top_p = st.sidebar.slider("top_p", 0.0, 1.0, self.generation_config.get('top_p', 1.0), step=0.01)
        temperature = st.sidebar.slider("temperature", 0.0, self.max_temperature, self.generation_config.get('temperature', 1.0), step=0.01)
        repetition_penalty = st.sidebar.slider("repetition_penalty", 0.0, self.max_repetition_penalty, self.generation_config.get('repetition_penalty', 1.0), step=0.1)
        buttonClean = st.sidebar.button("Clear history", key="clean")
        if buttonClean:
            st.session_state.history = []
            st.session_state.states = None
            cuda_empty_cache()
            st.rerun()
        
        system = st.sidebar.text_area(
            label="System Prompt (If exists)",
            height=200,
            value="",
        )
        functions = st.sidebar.text_area(
            label="Functions Json Format (If exists)",
            height=200,
            value="",
        )

        try:
            if functions is not None and functions.strip() != '':
                functions = json.loads(functions)
            else:
                functions = None
        except json.JSONDecodeError:
            functions = None
            log_warn('Functions implement not json format')

        self.system = system

        for i, message in enumerate(st.session_state.history):
            role = message['role']
            if role not in {'user', 'assistant'}:
                continue
            with st.chat_message(name=role, avatar=role):
                st.markdown(message.get('raw_content', message['content']))
        
        with st.chat_message(name="user", avatar="user"):
            input_placeholder = st.empty()
        with st.chat_message(name="assistant", avatar="assistant"):
            message_placeholder = st.empty()

        query = st.chat_input("请输入您的问题")
        if query:
            if query.strip() == "":
                st.warning('Input message could not be empty!', icon="⚠️")
            else:
                input_placeholder.markdown(query)
                history = st.session_state.history
                states = st.session_state.states
                self.generation_config['max_length'] = max_length
                self.generation_config['top_p'] = top_p
                self.generation_config['temperature'] = temperature
                self.generation_config['repetition_penalty'] = repetition_penalty
                self.generation_config['states'] = states

                input_text = self.build_prompt(query, history, functions)
                for response in self.model.stream_generate(input_text, **self.generation_config):
                    response = self.process_response_history(response, history)
                    message_placeholder.markdown(history[-1].get('raw_content', response))
                st.session_state.history = history
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
    content: Union[str, List[Dict]]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict] = None


class ChatCompletionRequest(BaseModel):
    model: str = 'default'
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length', 'function_call', None]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


OPENAI_START_DOCSTRING = r"""
    部署类似OpenAi的api server端

    #
    :param checkpoint_path: str, 模型所在的文件夹地址
    :param model_name: str, 模型名称
    :param generation_config: dict, 模型generate的参数设置

    # openai api参数
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    :param offload_when_nocall: str, 在一定时长内无调用就卸载模型，可以卸载到内存和disk
    :param offload_max_callapi_interval: int, 最长调用间隔
    :param offload_scheduler_interval: int, 定时任务的执行间隔
    :param api_keys: List[str], api keys的list
"""

@add_start_docstrings(OPENAI_START_DOCSTRING)
class ChatOpenaiApi(ChatBase):
    """
    TODO:
    1. 在后续调用服务，模型从cpu转到cuda上时，linux环境下内存下降，显存不能完全降低为0
    2. 【已修复】偶然会发生调用的时候，主线程和定时线程打架，导致device不一致的错误，因为forward过程时候发生offload
    3. cpu和delete测试通过，但是如何offload到disk上，不占用内存和显存
    """
    def __init__(self, checkpoint_path:str, model_name:str='default', route_api:str='/chat/completions', route_models:str='/models', 
                 offload_max_callapi_interval:int=24*3600, offload_scheduler_interval:int=10*60, offload_when_nocall:Literal['cpu', 'disk', 'delete']=None, 
                 api_keys:List[str]=None, **kwargs):
        assert kwargs.get('system') is None, "Args `system` is used in request key `message`"
        super().__init__(checkpoint_path, **kwargs)
        if not is_fastapi_available():
            raise ModuleNotFoundError("No module found, use `pip install fastapi`")
        from sse_starlette.sse import EventSourceResponse
        import sse_starlette
        if version.parse(sse_starlette.__version__) > version.parse('1.8'):
            log_warn('Module `sse_starlette` above 1.8 not support stream output, use `pip install sse_starlette==1.6.5`')
        self.offload_when_nocall = offload_when_nocall
        if offload_max_callapi_interval <= offload_scheduler_interval:
            raise ValueError('Args `offload_scheduler_interval` must < `offload_max_callapi_interval`')
        self.offload_max_callapi_interval = offload_max_callapi_interval  # 最长调用间隔
        self.offload_scheduler_interval = offload_scheduler_interval
        self.api_keys = api_keys
        self.EventSourceResponse = EventSourceResponse
        self.model_name = model_name
        self.role_user = 'user'
        self.role_assistant = 'assistant'

        if offload_when_nocall is None:
            self.app = FastAPI(lifespan=lifespan)
            self.lock = NoopContextManager()  # 仅用于占位，规整代码
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
        scheduler.add_job(self.check_call_and_offload, 'interval', seconds=self.offload_scheduler_interval)
        scheduler.start()
        self.app.state.scheduler = scheduler  # 将调度器存储在app的状态中，以便在shutdown时使用  
    
    def shutdown_event(scheduler):  
        if scheduler:  
            scheduler.shutdown()

    async def list_models(self):
        model_card = ModelCard(id=self.model_name)
        return ModelList(data=[model_card])

    def prepare_build_prompt_args(self, request):
        '''准备build_prompt的参数'''
        query = request.messages[-1].content
        history = [{'role': item.role, 'content': item.content} for item in request.messages[:-1]]
        input_args_or_kwargs = self.build_prompt(query, history, request.functions)
        return input_args_or_kwargs, history

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.model != self.model_name:
            raise HTTPException(status_code=404, detail=f"Invalid model: request.model:{request.model} != self.model_name:{self.model_name}")
        if request.messages[-1].role != self.role_user:  # 最后一条msg的role必须是user
            raise HTTPException(status_code=400, detail=f"Invalid request: messages last role shold be {self.role_user}")

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

        # 准备build_prompt的参数
        input_args_or_kwargs, history = self.prepare_build_prompt_args(request)

        with self.lock:
            self.model = self.build_model(**self.kwargs)
            self.last_callapi_timestamp = time.time()

            # 流式输出
            if request.stream:
                generate = self.predict(input_args_or_kwargs, request.model, history)
                return self.EventSourceResponse(generate, media_type="text/event-stream")
            
            # 非流式输出
            else:
                if isinstance(input_args_or_kwargs, (str,list)):
                    response = self.model.generate(input_args_or_kwargs, **self.generation_config)
                else:
                    response = self.model.generate(**input_args_or_kwargs, **self.generation_config)
                response = self.process_response_history(response, history)
                function_call = history[-1].get('function_call', None)
                choice_data = ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=self.role_assistant, content=response, function_call=function_call),
                    finish_reason= "function_call" if function_call is not None else "stop"
                )

                return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, input_args_or_kwargs: str, model_id: str, history:list):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=self.role_assistant),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        current_length = 0

        def get_generator(query):
            if isinstance(query, (str,list)):
                return self.model.stream_generate(query, **self.generation_config)
            else:
                return self.model.stream_generate(**query, **self.generation_config)
    
        for new_response in get_generator(input_args_or_kwargs):
            if len(new_response) == current_length:
                continue

            self.process_response_history(new_response, history)
            new_text = new_response[current_length:]
            current_length = len(new_response)

            function_call = history[-1].get('function_call', None)
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text, function_call=function_call),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        function_call = history[-1].get('function_call', None)
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(function_call=function_call),
            finish_reason= "function_call" if function_call is not None else "stop"
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        yield '[DONE]'

    def check_call_and_offload(self):
        '''检测距离上一次调用超出规定时间段，超出间隔则offload'''
        now = time.time()
        if not hasattr(self, 'model')  or (self.model is None):
            return
        elif not hasattr(self, 'last_callapi_timestamp'):
            self.last_callapi_timestamp = now
        elif now - self.last_callapi_timestamp > self.offload_max_callapi_interval:  # 超出调用间隔
            cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            if (self.offload_when_nocall == 'cpu') and (str(self.model.device) != 'cpu'):
                with self.lock:
                    # 如果没有调用，将模型转移到CPU
                    self.model.to('cpu')
                    log_free(f"{cur} - Model moved to cpu due to no activity for {self.offload_max_callapi_interval} sec.", prefix='[OFFLOAD]', prefix_color='cyan')
            elif (self.offload_when_nocall == 'disk') and hasattr(self, 'model'):
                with self.lock:
                    self.model = None
                    del self.model
                    log_free(f"{cur} - Model moved to disk due to no activity for {self.offload_max_callapi_interval} sec.", prefix='[OFFLOAD]', prefix_color='cyan')
            gc.collect()
            cuda_empty_cache()
        # print(now - self.last_callapi_timestamp)

    def run(self, app:str=None, host:str="0.0.0.0", port:int=8000, **kwargs):
        '''主程序入口'''
        import uvicorn
        uvicorn.run(app or self.app, host=host, port=port, **kwargs)


# ==========================================================================================
# =========================              各个具体模型实现        ============================
# ==========================================================================================
@add_start_docstrings(CHAT_START_DOCSTRING)
class Glm(ChatBase):
    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        # 没有system和function call
        if functions is not None:
            log_warn('ChatGlm do not support function call')
        
        if not history:
            prompt = query
        else:
            prompt, turn_i = "", 0
            if self.no_history_states():
                for query_or_response in history:
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


@add_start_docstrings(CHAT_START_DOCSTRING)
class Glm2(ChatBase):
    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('ChatGlm2 do not support function call')

        # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
        prompt, turn_i = "", 1
        if self.no_history_states():
            for query_or_response in history:
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


@add_start_docstrings(CHAT_START_DOCSTRING)
class Glm3(ChatBase):
    ''' functions格式如下:
    ```python
    [
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
    ```
    '''
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> list:
        if (len(history) == 0) or (history[0]["role"] != "system"):
            # 增加system信息
            if self.system:
                history.insert(0, {"role": "system", "content": self.system})
            elif functions is not None:
                history.insert(0, {
                    "role": "system", 
                    "content": "Answer the following questions as best as you can. You have access to the following tools:",
                    "tools": functions
                })

        # 增加tools信息
        if (functions is not None) and all(['tools' not in h for h in history]):
            history[0]['tools'] = functions

        if self.no_history_states():
            # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
            input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
        else:
            input_ids += self.generation_config['states']['last_token']
        history = self.update_history(history, query)
        return input_ids
        
    def process_response_history(self, response:str, history:List[dict]):
        response = super().process_response_history(response, history)
        if (not response) or (response[-1] == "�"):
            return response

        content = ""
        for resp in response.split("<|assistant|>"):
            if "\n" in resp:
                metadata, content = resp.split("\n", maxsplit=1)
            else:
                metadata, content = "", resp
            
            metadata = metadata.strip()
            content = content.strip().replace("[[训练时间]]", "2023年")
            raw_content = resp.strip().replace("[[训练时间]]", "2023年")  # 可用于cli，web_demo展示
            history[-1] = {"role": "assistant", "metadata": metadata, "content": content, "raw_content": raw_content}
            # 有functions
            if metadata and history[0]["role"] == "system" and "tools" in history[0]:
                try:
                    # 使用tools时候，stream_generate会有问题，因为中间结果是无法结构化解析的
                    def tool_call(**kwargs):
                        return kwargs
                    parameters = eval("\n".join(content.split("\n")[1:-1]))
                    history[-1]['function_call'] = {"name": metadata, "parameters": parameters}
                except SyntaxError:
                    pass
        return content


@add_start_docstrings(CHAT_START_DOCSTRING)
class Glm4(ChatBase):
    '''functions格式如下:
    ```python
    [
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
    ```
    '''
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None):
        if (len(history) == 0) or (history[0]["role"] != "system"):
            # 增加system信息
            if self.system:
                history.insert(0, {"role": "system", "content": self.system})
            elif functions is not None:
                history.insert(0, {"role": "system", "tools": functions, "content": ""})

        # 增加tools信息
        if (functions is not None) and all(['tools' not in h for h in history]):
            history[0]['tools'] = functions

        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        history = self.update_history(history, query)
        if self.no_history_states():
            prompt = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        else:
            prompt += self.generation_config['states']['last_token']
        return prompt
    
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
            
            metadata = metadata.strip()
            content = content.strip().replace("[[训练时间]]", "2024年")
            raw_content = resp.strip().replace("[[训练时间]]", "2024年")  # 可用于cli，web_demo展示
            history[-1] = {"role": "assistant", "metadata": metadata, "content": content, "raw_content": raw_content}
            # 有functions        
            if metadata and history[0]["role"] == "system" and "tools" in history[0]:
                history[-1]['function_call'] = {"name": metadata.strip(), "parameters": content}
        return content


@add_start_docstrings(CHAT_START_DOCSTRING)
class InternLM(ChatBase):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else SYSTEM_ZH

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None):
        # InternLM v1不支持function call
        if functions is not None: 
            log_warn('InternLM do not support function call')
        if self.tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = self.tokenizer.bos_token
        
        if self.no_history_states():
            prompt += f"""<|System|>:{self.system}\n"""
            for query_or_response in history:
                if query_or_response['role'] == 'user':
                    prompt += f"""<|User|>:{query_or_response['content']}\n"""
                elif query_or_response['role'] == 'assistant':
                    prompt += f"""<|Bot|>:{query_or_response['content']}<eoa>\n"""
        else:
            prompt += self.generation_config['states']['last_token']

        prompt += f"""<|User|>:{query}\n<|Bot|>:"""
        history = self.update_history(history, query)
        return prompt

    def process_response_history(self, response, history=None):
        response = response.split("<eoa>")[0]
        response = super().process_response_history(response, history)
        return response


@add_start_docstrings(CHAT_START_DOCSTRING)
class InternLM2(ChatBase):
    '''internlm2支持function call, 格式如下:

    由于_additional_special_tokens为['<|im_start|>', '<|im_end|>', '<|action_start|>', '<|action_end|>', '<|interpreter|>', '<|plugin|>']
    在function call时候若skip_special_tokens=True, 则捕捉不到'<|action_start|>', '<|action_end|>', '<|interpreter|>', '<|plugin|>'
    因此bert4torch_config.json中未设置skip_special_tokens, 默认为False
    '''
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else SYSTEM_ZH
        self.plugin_with_name = True

        self.api_prefix = (
            "This is the subfunction for tool '{tool_name}', you can use this tool. "
            'The description of this function is: \n{description}')

        self.meta_prompt = ('当开启工具以及代码时，根据需求选择合适的工具进行调用')

        INTERPRETER_CN = ('你现在已经能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。'
                        '当你向 python 发送含有 Python 代码的消息时，它将在该环境中执行。'
                        '这个工具适用于多种场景，如数据分析或处理（包括数据操作、统计分析、图表绘制），'
                        '复杂的计算问题（解决数学和物理难题），编程示例（理解编程概念或特性），'
                        '文本处理和分析（比如文本解析和自然语言处理），'
                        '机器学习和数据科学（用于展示模型训练和数据可视化），'
                        '以及文件操作和数据导入（处理CSV、JSON等格式的文件）。')

        self.plugin_prompt = ('你可以使用如下工具：'
                    '\n{prompt}\n'
                    '如果你已经获得足够信息，请直接给出答案. 避免不必要的工具调用! '
                    '同时注意你可以使用的工具，不要随意捏造！')

    
    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None):
        if (len(history) == 0) or (history[0]["role"] != "system"):
            history.insert(0, {"role": "system", "content": self.system if functions is None else self.meta_prompt})

        if (functions is not None) and all([h['role'] !='function' for h in history]):
            # history中没有function
            start = [i for i, v in enumerate(history) if v['role']=='system'][-1] + 1
            plugin_descriptions = []
            for i, func in enumerate(functions, start=start):
                plugin = copy.deepcopy(func)
                name = plugin['name'].split('.')[0]
                plugin['description'] = self.api_prefix.format(tool_name=name, description=plugin['description'])
                plugin_descriptions.append(plugin)
            
            plugin_prompt = self.plugin_prompt.format(prompt=json.dumps(plugin_descriptions, ensure_ascii=False, indent=4))
            
            if self.plugin_with_name:
                content = f"""<|im_start|>system name=<|plugin|>\n{plugin_prompt}<|im_end|>\n"""
            else:
                content = f"""<|im_start|>system\n<|plugin|>\n{plugin_prompt}<|im_end|>\n"""
            history.insert(i, {"role": "function", "content": content})

        if self.tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = self.tokenizer.bos_token
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if role == 'system':
                    prompt += f"""<|im_start|>system\n{content}<|im_end|>\n"""
                elif role == 'function':
                    prompt += content
                elif role == 'user':
                    prompt += f"""<|im_start|>user\n{content}<|im_end|>\n"""
                elif role == 'assistant':
                    prompt += f"""<|im_start|>assistant\n{content}<|im_end|>\n"""
        else:
            prompt += self.generation_config['states']['last_token']

        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        history = self.update_history(history, query)
        return prompt

    def process_response_history(self, response, history=None):
        response = response.split("<|im_end|>")[0]
        for key in ['<|im_start|>', '<|im_end|>']:
            response = response.replace(key, '')
        response = super().process_response_history(response, history)

        start_token = '<|action_start|>'
        end_token = '<|action_end|>'
        for _token in ['<|plugin|>', '<|interpreter|>']:
            if _token in response:
                response, arguments = response.split(f"{start_token}{_token}")
                arguments = arguments.split(end_token)[0].strip()
                response = response.split(start_token)[0]
                history[-1]['function_call'] = {"name": 'plugin', "arguments": arguments}

        return response


@add_start_docstrings(CHAT_START_DOCSTRING)
class Qwen(ChatBase):
    '''functions格式如下:
    ```python
    [
        {
            'name_for_human': '谷歌搜索',
            'name_for_model': 'google_search',
            'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Format the arguments as a JSON object.',
            'parameters': [{
                'name': 'search_query',
                'description': '搜索关键词或短语',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
    ]
    '''
    def __init__(self, *args, system:str=None, max_window_size=6144, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else 'You are a helpful assistant.'
        self.max_window_size = max_window_size
        self.observation_ids = self.tokenizer.encode('Observation:')

    def parse_messages(self, query, messages, functions):
        '''copy from https://github.com/QwenLM/Qwen/blob/main/openai_api.py'''

        TOOL_DESC = (
            '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
            ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
        )

        REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

        {tools_text}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tools_name_text}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!"""

        messages = copy.deepcopy(messages)
        if len(messages) > 0 and messages[0]['role'] == 'system':
            system = messages.pop(0)['content'].lstrip('\n').rstrip()
        else:
            system = self.system

        if functions:
            tools_text = []
            tools_name_text = []
            for func_info in functions:
                name = func_info.get('name', '')
                name_m = func_info.get('name_for_model', name)
                name_h = func_info.get('name_for_human', name)
                desc = func_info.get('description', '')
                desc_m = func_info.get('description_for_model', desc)
                tool = TOOL_DESC.format(
                    name_for_model=name_m,
                    name_for_human=name_h,
                    # Hint: You can add the following format requirements in description:
                    #   "Format the arguments as a JSON object."
                    #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                    description_for_model=desc_m,
                    parameters=json.dumps(func_info['parameters'], ensure_ascii=False),
                )
                tools_text.append(tool)
                tools_name_text.append(name_m)
            tools_text = '\n\n'.join(tools_text)
            tools_name_text = ', '.join(tools_name_text)
            instruction = (REACT_INSTRUCTION.format(
                tools_text=tools_text,
                tools_name_text=tools_name_text,
            ).lstrip('\n').rstrip())
        else:
            instruction = ''

        messages_with_fncall = messages
        messages = []
        for m_idx, m in enumerate(messages_with_fncall):
            role, content, func_call = m['role'], m['content'], m.get('function_call')
            content = content or ''
            content = content.lstrip('\n').rstrip()
            if role == 'function':
                if (len(messages) == 0) or (messages[-1]['role'] != 'assistant'):
                    raise ValueError('Expecting role assistant before role function.')
                messages[-1]['content'] += f'\nObservation: {content}'
                if m_idx == len(messages_with_fncall) - 1:
                    # add a prefix for text completion
                    messages[-1]['content'] += '\nThought:'
            elif role == 'assistant':
                if len(messages) == 0:
                    raise ValueError('Expecting role user before role assistant.')
                if func_call is None:
                    if functions:
                        content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
                else:
                    f_name, f_args = func_call['name'], func_call['arguments']
                    if not content.startswith('Thought:'):
                        content = f'Thought: {content}'
                    content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
                if messages[-1]['role'] == 'user':
                    messages.append({'role': 'assistant', 'content': content.lstrip('\n').rstrip()})
                else:
                    messages[-1]['content'] += '\n' + content
            elif role == 'user':
                messages.append({'role': 'user', 'content': content.lstrip('\n').rstrip()})
            else:
                raise ValueError(f'Incorrect role {role}.')

        if len(messages) % 2 != 0:
            raise ValueError(f'{messages} len = {len(messages)}, not paired')

        history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
        for i in range(0, len(messages), 2):
            if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
                usr_msg = messages[i]['content'].lstrip('\n').rstrip()
                bot_msg = messages[i + 1]['content'].lstrip('\n').rstrip()
                if instruction and (i == len(messages) - 2):
                    usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                    instruction = ''
                history.append([usr_msg, bot_msg])
            else:
                raise ValueError('Expecting exactly one user (or function) role before every assistant role.')
        if instruction:
            query = f'{instruction}\n\nQuestion: {query}'
        return query, history, system

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions:
            # 如果使用了functions则需要增加eos_token_id
            if self.observation_ids not in self.generation_config.get('eos_token_id', []):
                self.generation_config['eos_token_id'] = self.generation_config.get('eos_token_id', []) + [self.observation_ids]

        instruction_query, history_list, system = self.parse_messages(query, history, functions)
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", system)
        raw_text = ""

        if self.no_history_states():
            for turn_query, turn_response in reversed(history_list):
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

        raw_text += f"\n{im_start}user\n{instruction_query}{im_end}\n{im_start}assistant\n"
        history = self.update_history(history, query)
        return raw_text

    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        response = super().process_response_history(response, history)
        func_name, func_args = '', ''
        i = response.find('\nAction:')
        j = response.find('\nAction Input:')
        k = response.find('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is omitted by the LLM,
                # because the output text may have discarded the stop word.
                response = response.rstrip() + '\nObservation:'  # Add it back.
            k = response.find('\nObservation:')
            func_name = response[i + len('\nAction:'):j].strip()
            func_args = response[j + len('\nAction Input:'):k].strip()

        if func_name:
            response = response[:i]
            t = response.find('Thought: ')
            if t >= 0:
                response = response[t + len('Thought: '):]
            response = response.strip()
            try:
                json.loads(func_args)
                history[-1]['function_call'] = {'name': func_name, 'arguments': func_args}
            except json.JSONDecodeError:
                pass
        else:
            z = response.rfind('\nFinal Answer: ')
            if z >= 0:
                response = response[z + len('\nFinal Answer: '):]
        history[-1]['content'] = response
        # if 'Observation:' in history[-1]['raw_content']:
        #     history[-1]['raw_content'] = history[-1]['raw_content'].replace('Observation:', '')
        return response


@add_start_docstrings(CHAT_START_DOCSTRING)
class Qwen2(ChatBase):
    '''Qwen2的chat, 含function call的逻辑
    主要参考了qwen_agent的逻辑
    :param parallel_function_calls: bool, 允许并行调用function tools工具

    ### functions格式如下:
    ```python
    [
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。 Format the arguments as a JSON object.',
            'parameters': [{
                'name': 'prompt',
                'description': '英文关键词，描述了希望图像具有什么内容',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
    ]
    '''
    def __init__(self, *args, system:str=None, parallel_function_calls:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else 'You are a helpful assistant.'
        self.parallel_function_calls = parallel_function_calls

        self.FN_NAME = '✿FUNCTION✿'
        self.FN_ARGS = '✿ARGS✿'
        self.FN_RESULT = '✿RESULT✿'
        self.FN_EXIT = '✿RETURN✿'
        self.FN_STOP_WORDS = [self.FN_RESULT, self.FN_EXIT]
        self.FN_STOP_WORDS_IDS = [self.tokenizer.encode(i) for i in self.FN_STOP_WORDS]

        FN_CALL_TEMPLATE_INFO_ZH = """# 工具

        ## 你拥有如下工具：

        {tool_descs}"""

        FN_CALL_TEMPLATE_INFO_EN = """# Tools

        ## You have access to the following tools:

        {tool_descs}"""

        FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

        %s: 工具名称，必须是[{tool_names}]之一。
        %s: 工具输入
        %s: 工具结果
        %s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_RESULT,
            self.FN_EXIT,
        )

        FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

        %s: The tool to use, should be one of [{tool_names}]
        %s: The input of the tool
        %s: Tool results
        %s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_RESULT,
            self.FN_EXIT,
        )

        FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

        %s: 工具1的名称，必须是[{tool_names}]之一
        %s: 工具1的输入
        %s: 工具2的名称
        %s: 工具2的输入
        ...
        %s: 工具N的名称
        %s: 工具N的输入
        %s: 工具1的结果
        %s: 工具2的结果
        ...
        %s: 工具N的结果
        %s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_RESULT,
            self.FN_RESULT,
            self.FN_RESULT,
            self.FN_EXIT,
        )

        FN_CALL_TEMPLATE_FMT_PARA_EN = """## Insert the following command in your reply when you need to call N tools in parallel:

        %s: The name of tool 1, should be one of [{tool_names}]
        %s: The input of tool 1
        %s: The name of tool 2
        %s: The input of tool 2
        ...
        %s: The name of tool N
        %s: The input of tool N
        %s: The result of tool 1
        %s: The result of tool 2
        ...
        %s: The result of tool N
        %s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_NAME,
            self.FN_ARGS,
            self.FN_RESULT,
            self.FN_RESULT,
            self.FN_RESULT,
            self.FN_EXIT,
        )

        self.FN_CALL_TEMPLATE = {
            'zh': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_ZH,
            'en': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_EN,
            'zh_parallel': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_ZH,
            'en_parallel': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_EN,
        }
        
    @staticmethod
    def get_function_description(function: Dict, lang: Literal['en', 'zh']) -> str:
        """
        Text description of function  # copy from qwen_agent
        """
        tool_desc_template = {
            'zh': '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
            'en': '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
        }
        tool_desc = tool_desc_template[lang]
        name = function.get('name', None)
        name_for_human = function.get('name_for_human', name)
        name_for_model = function.get('name_for_model', name)
        assert name_for_human and name_for_model

        if name_for_model == 'code_interpreter':
            args_format = {
                'zh': '此工具的输入应为Markdown代码块。',
                'en': 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
            }
        else:
            args_format = {
                'zh': '此工具的输入应为JSON对象。',
                'en': 'Format the arguments as a JSON object.',
            }
        args_format = function.get('args_format', args_format[lang])

        return tool_desc.format(name_for_human=name_for_human,
                                name_for_model=name_for_model,
                                description_for_model=function.get('description', function['description_for_model']),
                                parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                args_format=args_format).rstrip()

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions:
            # 如果使用了functions则需要增加eos_token_id
            if all([i not in self.generation_config.get('eos_token_id', []) for i in self.FN_STOP_WORDS_IDS]):
                self.generation_config['eos_token_id'] = self.generation_config.get('eos_token_id', []) + self.FN_STOP_WORDS_IDS

        if (len(history) == 0) or (history[0]["role"] != "system"):
            history.insert(0, {"role": "system", "content": self.system})
        
        # 处理functions的逻辑, copy from qwen_agent
        if (functions is not None) and all(['tools' not in h for h in history]):
            lang = 'en'
            for m in history:
                if has_chinese_char(m['content']):
                    lang = 'zh'
                    break

            tool_desc_template = self.FN_CALL_TEMPLATE[lang + ('_parallel' if self.parallel_function_calls else '')]
            tool_descs = '\n\n'.join(self.get_function_description(function, lang=lang) for function in functions)
            tool_names = ','.join(function.get('name', function.get('name_for_model', '')) for function in functions)
            tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)
            history[0]['content'] += '\n\n' + tool_system
            history[0]['tools'] = tool_system  # 仅用于是否已经添加过functions的判断

        history = self.update_history(history, query)
        if self.no_history_states():
            prompt = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        else:
            prompt += self.generation_config['states']['last_token']

        return prompt
    
    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        """
        If the model calls function by built-in function call template,
        convert and display it in function_call format.
        """
        response = super().process_response_history(response, history)

        # Remove ': ' brought by continued generation of function calling
        if response.startswith(': '):
            response = response[2:]
        elif response.startswith(':'):
            response = response[1:]

        i = response.find(f'{self.FN_NAME}:')

        # 没有function call
        if i < 0:
            show_text = self.remove_incomplete_special_tokens(response)
            return show_text

        # 在function call前说了部分描述
        thought = None
        if i > 0:
            answer = response[:i].lstrip('\n').rstrip()
            if answer.endswith('\n'):
                answer = answer[:-1]
            thought = self.remove_incomplete_special_tokens(answer)
            # if thought:
            #     history[-1]['content'] = thought
            response = response[i:]

        # 有function call
        for part in response.split(f'{self.FN_NAME}:'):
            if not part:
                continue
            if part.endswith('\n'):
                part = part[:-1]

            arg_sep = f'\n{self.FN_ARGS}:'
            i = part.find(arg_sep)
            if i < 0:
                fn_name = part.strip()
                list_of_fn_args = ['']
            else:
                fn_name = part[:i].strip()
                list_of_fn_args = [_.strip() for _ in part[i + len(arg_sep):].split(arg_sep)]
            fn_name = self.remove_incomplete_special_tokens(fn_name)
            for fn_args in list_of_fn_args:
                fn_args = self.remove_incomplete_special_tokens(fn_args)
                fn_args = self.remove_trailing_comment_of_fn_args(fn_args)
                history[-1]['function_call'] = {'name': fn_name, 'arguments': fn_args}
        return thought or response
    
    def remove_incomplete_special_tokens(self, text: str) -> str:
        special_tokens = (self.FN_NAME, self.FN_ARGS, self.FN_RESULT, self.FN_EXIT)
        text = text.rstrip()
        if text.endswith(special_tokens):
            for s in special_tokens:
                if text.endswith(s):
                    text = text[:-len(s)]
                    break
        else:
            trail_start = text.rfind('✿')
            trail_token = text[trail_start:]
            for s in special_tokens:
                if s.startswith(trail_token):
                    text = text[:trail_start]
                    break
        text = text.lstrip('\n').rstrip()
        return text
    
    @staticmethod
    def remove_trailing_comment_of_fn_args(fn_args: str):
        fn_args = fn_args.strip()

        if fn_args.startswith('{'):
            k = fn_args.rfind('}')
            if k > 0:
                fn_args = fn_args[:k + 1]

        if fn_args.startswith('```'):
            k = fn_args.rfind('\n```')
            if k > 0:
                fn_args = fn_args[:k + 4]
        return fn_args


@add_start_docstrings(CHAT_START_DOCSTRING)
class LLaMA2(ChatBase):
    '''LLaMA2
    LLaMA由于只有base模型, 没有chat所以直接model.generate即可
    '''
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('LLaMA2 do not support function call')

        if (len(history) == 0) or (history[0]["role"] != "system"):
            system = self.system or SYSTEM_ZH if has_chinese_char(query) else SYSTEM_EN
            history.insert(0, {"role": "system", "content": system})

        texts = ''
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content'].strip()
                if role == 'system':
                    texts += f'[INST] <<SYS>>\n{content}\n<</SYS>>\n\n'
                elif role == 'user':
                    texts += f'{content} [/INST] '
                elif role == 'assistant':
                    texts += f"{content} </s><s> [INST] "
        else:
            texts = self.generation_config['states']['last_token']

        texts += f'{query.strip()} [/INST]'
        history = self.update_history(history, query)
        return texts


@add_start_docstrings(CHAT_START_DOCSTRING)
class ApplyChatTemplate(ChatBase):
    '''直接使用self.tokenizer.apply_chat_template来构建输入
    如果模型直接沿用这种方式，则无需做特殊的处理
    '''
    def __init__(self, *args, system:str=None, add_generation_prompt:bool=True, 
                 tokenize:bool=False, tools_in_user_message:bool=False, 
                 enable_thinking:bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
        self.apply_chat_template_config = {
            "add_generation_prompt": add_generation_prompt, 
            "tokenize": tokenize,
            "tools_in_user_message": tools_in_user_message,
            "enable_thinking": enable_thinking
        }

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if self.system and ((len(history) == 0) or (history[0]["role"] != "system")):
            history.insert(0, {"role": "system", "content": self.system})

        history = self.update_history(history, query)
        if self.no_history_states():
            # llama3.1支持function call
            tools = functions
            if (functions is not None) and (not isinstance(functions, list)):
                tools = [functions]
            texts = self.tokenizer.apply_chat_template(history, tools = tools, **self.apply_chat_template_config)
        else:
            texts = self.generation_config['states']['last_token']
            texts += f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        return texts
    
    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        response = super().process_response_history(response, history)
        try:
            json.loads(response)
            history[-1]['function_call'] = response
        except json.JSONDecodeError:
            pass
        return response

@add_start_docstrings(CHAT_START_DOCSTRING)
class LLaMA3(ApplyChatTemplate):
    '''llama3不支持function call, llama3.1支持function call
    
    ### LLaMA3.1请求的Example
    ```json
    [
        {"role": "system", "content": "You are a bot that responds to weather queries."},
        {"role": "user", "content": "Hey, what's the temperature in Paris right now?"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]},
        {"role": "tool", "name": "get_current_temperature", "content": "22.0"}
    ]
    ```
    '''
    pass


@add_start_docstrings(CHAT_START_DOCSTRING)
class DeepSeekR1(ApplyChatTemplate):
    '''直接使用self.tokenizer.apply_chat_template来构建输入
    如果模型直接沿用这种方式，则无需做特殊的处理
    '''
    def process_response_history(self, response:Union[str,tuple,list], history:List[dict]=None) -> str:
        response = super().process_response_history(response, history)
        if '</think>' in response:
            history[-1]['content'] = response.split('</think>')[-1].strip()
        return response


@add_start_docstrings(CHAT_START_DOCSTRING)
class Ziya(ChatBase):
    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('Ziya do not support function call')

        prompt = ''
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if role == 'user':
                    prompt += f"<human>:{content}\n"
                elif role == 'assistant':
                    prompt += f"<bot>:{content}\n"
        else:
            prompt += self.generation_config['states']['last_token']
        
        prompt += f"<human>:{query.strip()}\n<bot>:"
        history = self.update_history(history, query)
        return prompt


@add_start_docstrings(CHAT_START_DOCSTRING)
class ChineseLlamaAlpaca(ChatBase):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system or \
("Below is an instruction that describes a task. "
"Write a response that appropriately completes the request.\n\n"
)

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('ChineseAlphaLLaMA do not support function call')

        if (len(history) == 0) or (history[0]["role"] != "system"):
            history.insert(0, {"role": "system", "content": self.system})

        prompt = ''
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if role == 'system':
                    prompt = self.system + prompt
                elif role == 'user':
                    prompt += f"### Instruction:\n\n{content}\n\n"
                elif role == 'assistant':
                    prompt += f"### Response:\n\n{content}\n\n"
            prompt += f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        else:
            prompt += self.generation_config['states']['last_token'] + f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        
        history = self.update_history(history, query)
        return prompt


@add_start_docstrings(CHAT_START_DOCSTRING)
class Belle(ChatBase):
    def build_tokenizer(self, **kwargs):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.checkpoint_path, use_fast=False)
    
    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('Belle do not support function call')

        prompt = ''
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if role == 'user':
                    prompt += f"Human: {content} \n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
        else:
            prompt += self.generation_config['states']['last_token']
        prompt += f"Human: {query} \n\nAssistant: "
        history = self.update_history(history, query)
        return prompt


@add_start_docstrings(CHAT_START_DOCSTRING)
class Baichuan(ChatBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_token_id = kwargs.get('user_token_id', 195)
        self.assistant_token_id = kwargs.get('assistant_token_id', 196)

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('Baichuan do not support function call')

        total_input = []
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if role == 'user':
                    total_input += [self.user_token_id] + self.tokenizer.encode(content)
                elif role == 'assistant':
                    total_input += [self.assistant_token_id] + self.tokenizer.encode(content) + [self.tokenizer.eos_token_id]
        else:
            total_input += [self.generation_config['states']['last_token_id']]
        total_input += [self.user_token_id] + self.tokenizer.encode(query) + [self.assistant_token_id]
        
        history = self.update_history(history, query)
        return total_input


@add_start_docstrings(CHAT_START_DOCSTRING)
class PretrainedTextContinuation(ChatBase):
    '''预训练的模型续写'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_prompt(self, query:str, history:List[dict], functions:List[dict]=None) -> str:
        if functions is not None: 
            log_warn('PretrainedTextContinuation do not support function call')

        total_input = ''
        if self.no_history_states():
            for query_or_response in history:
                role, content = query_or_response['role'], query_or_response['content']
                if self.generation_config.get('include_input', False):
                    if role == 'assistant':
                        total_input += content
                else:
                    total_input += content
        else:
            total_input += [self.generation_config['states']['last_token_id']]
        total_input += query
        
        history = self.update_history(history, query)
        return total_input


LLM_MAPPING = {
    'glm': Glm,
    'glm2': Glm2,
    'glm3': Glm3,
    'glm4': Glm4,
    'internlm': InternLM,
    'internlm2': InternLM2,
    'qwen': Qwen,
    'qwen2': Qwen2,
    'qwen3': ApplyChatTemplate,
    'qwen3_moe': ApplyChatTemplate,
    'llama2': LLaMA2,
    'llama3': LLaMA3,
    'ziya': Ziya,
    'chinese_llama_alpaca': ChineseLlamaAlpaca,
    'belle': Belle,
    'baichuan': Baichuan,
    'apply_chat_template': ApplyChatTemplate,
    'pretrained_text_continuation': PretrainedTextContinuation,
    'deepseek_r1': DeepSeekR1
}
