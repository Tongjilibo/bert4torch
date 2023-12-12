import json
import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from bert4torch.snippets import log_warn
import importlib
if importlib.util.find_spec("fastapi") is not None:
    from fastapi import FastAPI, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
if importlib.util.find_spec("sse_starlette") is not None:
    from sse_starlette.sse import ServerSentEvent, EventSourceResponse


class WebServing(object):
    """简单的Web接口，基于bottlepy简单封装，仅作为临时测试使用，不保证性能。

    Example:
        >>> arguments = {'text': (None, True), 'n': (int, False)}
        >>> web = WebServing(port=8864)
        >>> web.route('/gen_synonyms', gen_synonyms, arguments)
        >>> web.start()
        >>> # 然后访问 http://127.0.0.1:8864/gen_synonyms?text=你好
    
    依赖（如果不用 server='paste' 的话，可以不装paste库）:
        >>> pip install bottle
        >>> pip install paste
    """
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        """封装为接口函数

        :param func: 要转换为接口的函数，需要保证输出可以json化，即需要保证 json.dumps(func(inputs)) 能被执行成功；
        :param arguments: 声明func所需参数，其中key为参数名，value[0]为对应的转换函数（接口获取到的参数值都是字符串型），value[1]为该参数是否必须；
        :param method: 'GET'或者'POST'。
        """
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        """添加接口"""
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        """启动服务"""
        self.bottle.run(host=self.host, port=self.port, server=self.server)



# Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


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


class OpenAIApi():
    def __init__(self, model, tokenizer, generation_config=None, route_api='/chat', route_models='/models') -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or dict()
        router = APIRouter()
        self.app = FastAPI(lifespan=lifespan)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        router.add_api_route(route_models, endpoint=self.list_models, response_model=ModelList)
        self.app.include_router(router)
        router.add_api_route(route_api, endpoint=self.create_chat_completion, response_model=ChatCompletionResponse)

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        uvicorn.run(self.app, host=host, port=port, **kwargs)

    def build_prompt(message):
        raise NotImplementedError

    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            query = prev_messages.pop(0).content + query

        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                    raise HTTPException(status_code=400, detail='Arg `messages` do not follow user, assistant format')

        # 流式输出
        if request.stream:
            generate = self.predict(query, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")

        # 非流式输出
        messages = []
        for i in request.messages:
            messages.append({'role': i.role, 'content': i.content})
        
        input_text = self.build_prompt(messages)
        response = self.model.generate(input_text, tokenizer=self.tokenizer, **self.generation_config)
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )

        return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, query: str, model_id: str):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        current_length = 0

        for new_response in self.model.stream_generate(query, tokenizer=self.tokenizer, **self.generation_config):
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

