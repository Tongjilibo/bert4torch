'''
利用fastapi搭建openai格式的server和client调用
Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
Usage: python openai_api.py
Visit http://localhost:8000/docs for documents.
'''

import time
import json
import requests
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from bert4torch.snippets import log_info, log_warn, cuda_empty_cache, AnyClass
from bert4torch.snippets import is_fastapi_available, is_pydantic_available, is_sseclient_available
from packaging import version
from .base import Chat
import gc

FastAPI, BaseModel, Field= object, object, AnyClass
if is_fastapi_available():
    from fastapi import FastAPI, HTTPException, APIRouter, BackgroundTasks
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
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[int] = None


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
    def __init__(self, model_path, name='default', route_api='/chat/completions', route_models='/models', 
                 max_callapi_interval=24*3600, offload_when_nocall:Literal['cpu', 'disk']=None, **kwargs):
        self.offload_when_nocall = offload_when_nocall
        if offload_when_nocall is not None:
            kwargs['create_model_at_startup'] = False
        super().__init__(model_path, **kwargs)
        assert is_fastapi_available(), "No module found, use `pip install fastapi`"
        from sse_starlette.sse import ServerSentEvent, EventSourceResponse
        import sse_starlette
        if version.parse(sse_starlette.__version__) > version.parse('1.8'):
            log_warn('Module `sse_starlette` above 1.8 not support stream output')
        self.max_callapi_interval = max_callapi_interval  # 最长调用间隔
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
        
    def run(self, app:str=None, host:str="0.0.0.0", port:int=8000, **kwargs):
        import uvicorn
        uvicorn.run(app or self.app, host=host, port=port, **kwargs)

    async def list_models(self):
        model_card = _ModelCard(id=self.name)
        return _ModelList(data=[model_card])

    async def create_chat_completion(self, request: _ChatCompletionRequest, background_tasks: BackgroundTasks):
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
        self.model = self.build_model()
        
        # 后台任务
        if self.offload_when_nocall is not None:
            self.last_callapi_timestamp = time.time()
            background_tasks.add_task(self.check_last_call)

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

    def check_last_call(self):
        '''检测距离上一次调用超出规定时间段'''
        while True:
            now = time.time()
            if now - self.last_callapi_timestamp > self.max_callapi_interval:  # 超出调用间隔
                if (self.offload_when_nocall == 'cpu') and (str(self.model.device) != 'cpu'):
                    # 如果没有调用，将模型转移到CPU
                    self.model.to('cpu')  
                    log_info(f"Model moved to cpu due to no activity for {self.max_callapi_interval} sec.")
                    gc.collect()
                    cuda_empty_cache()
                elif (self.offload_when_nocall == 'disk') and hasattr(self, 'model'):
                    self.model = None
                    del self.model
                    log_info(f"Model moved to disk due to no activity for {self.max_callapi_interval} sec.")
                    gc.collect()
                    cuda_empty_cache()
            time.sleep(10*60)  # 每10min检查一次


class ChatOpenaiClient:
    '''使用openai来调用
    
    Example
    --------------------------------------------
    messages = [
            {"content": "你好", "role": "user"},
            {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
            {"content": "你可以做什么？", "role": "user"}
            ]
    client = ChatOpenaiClient('http://127.0.0.1:8000')
    # 流式
    for token in client.stream_chat(messages):
        print(token, end='', flush=True)
    # 非流式
    print(client.chat(messages))
    '''
    def __init__(self, base_url) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
    
    def stream_chat(self, messages, max_length=None, temperature=None, top_p=None):
        '''流式返回'''
        response = self.client.chat.completions.create(
            model='default',
            messages=messages,
            stream=True,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p
            )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    def chat(self, messages, max_length=None, temperature=None, top_p=None):
        '''一次性返回'''
        response = self.client.chat.completions.create(
            model='default',
            messages=messages,
            stream=False,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p
            )
        content = response.choices[0].message.content
        return content
    
    def stream_chat_cli(self, *args, **kwargs):
        for token in self.stream_chat(*args, **kwargs):
            print(token, end='', flush=True)


class ChatOpenaiClientSseclient:
    '''调用openai接口的client, 流式请求

    Example
    --------------------------------------------
    messages = [
            {"content": "你好", "role": "user"},
            {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
            {"content": "你可以做什么？", "role": "user"}
            ]
    client = ChatOpenaiClientSseclient('http://127.0.0.1:8000')
    # 测试打印
    client.stream_chat_cli(body)
    # 流式
    for token in client.stream_chat(body):
        print(token, end='', flush=True)
    '''
    def __init__(self, url) -> None:
        self.url = url
        if is_sseclient_available():
            import sseclient
        else:
            raise ImportError('No module found, you may `pip install sseclient-py`')
        
        self.sseclient = sseclient
   
    def stream_chat(self, body):
        '''接口调用'''
        reqHeaders = {'Accept': 'text/event-stream'}
        request = requests.post(self.url, stream=True, headers=reqHeaders, json=body)
        client = self.sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    yield data['content']

    def stream_chat_cli(self, body):
        '''简单测试在命令行打印'''
        for token in self.stream_chat(body):
            print(token, end='', flush=True)


def extend_with_chat_openai_api(InputModel):
    """添加ChatWebDemo"""
    class ChatDemo(InputModel, ChatOpenaiApi):
        pass
    return ChatDemo