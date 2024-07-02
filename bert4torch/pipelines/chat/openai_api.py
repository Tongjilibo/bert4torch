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
from bert4torch.snippets import log_info, log_warn, cuda_empty_cache
from bert4torch.snippets import is_fastapi_available, is_pydantic_available, is_sseclient_available
from packaging import version
from .base import Chat
import gc
import threading


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


__all__ = [
    'ChatOpenaiApi',
    'ChatOpenaiClient',
    'ChatOpenaiClientSseclient'
    ]

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
    model: str
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

    >>> checkpoint_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
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
        from sse_starlette.sse import ServerSentEvent, EventSourceResponse
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
        self.role_system = 'system'

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

        if request.messages[-1].role != self.role_user:
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        # 首条可以是role_system
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
        
        if self.offload_when_nocall is None:
            self.model = self._build_model()
        else:
            with self.lock:
                self.model = self._build_model()
            self.last_callapi_timestamp = time.time()

        # 流式输出
        if request.stream:
            generate = self.predict(input_text, request.model)
            return self.EventSourceResponse(generate, media_type="text/event-stream")
        
        # 非流式输出
        else:
            response = self.model.generate(input_text, **self.generation_config)
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role=self.role_assistant, content=response),
                finish_reason="stop"
            )

            return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, query: str, model_id: str):
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

            new_text = new_response[current_length:]
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
