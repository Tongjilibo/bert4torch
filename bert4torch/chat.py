import os
import torch
import time
import json
import requests
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from torch4keras.snippets import log_warn, log_warn_once
import importlib
if importlib.util.find_spec("uvicorn") is not None:
    import uvicorn
if importlib.util.find_spec("pydantic") is not None:
    from pydantic import BaseModel, Field
if importlib.util.find_spec("fastapi") is not None:
    from fastapi import FastAPI, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
if importlib.util.find_spec("sse_starlette") is not None:
    from sse_starlette.sse import ServerSentEvent, EventSourceResponse
if importlib.util.find_spec("sseclient") is not None:
    import sseclient


class Chat:
    '''聊天类'''
    def __init__(self, model_path, **generation_config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.checkpoint_path = model_path
        self.config_path = os.path.join(model_path, 'bert4torch_config.json')
        self.generation_config = generation_config
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.generation_config['tokenizer'] = self.tokenizer
        self.model = self.build_model()

    def build_prompt(self, query, history) -> str:
        raise NotImplementedError

    def build_model(self):
        from bert4torch.models import build_transformer_model
        model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path)
        model = model.half()
        model = model.to(self.device)
        return model
    
    def process_response(self, response):
        return response


class ChatCliDemo(Chat):
    '''在命令行中交互的demo'''
    def __init__(self, *args, **generation_config):
        super().__init__(*args, **generation_config)
        self.init_str = "输入内容进行对话，clear清空对话历史，stop终止程序"
        self.history_maxlen = 3

    def build_cli_text(self, history):
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
                    response = self.process_response(response)
                    new_history = history + [(query, response)]
                    os.system(clear_command)
                    print(self.build_cli_text(previous_history + new_history), flush=True)
            else:
                response = self.model.generate(prompt, **self.generation_config)
                response = self.process_response(response)
                new_history = history + [(query, response)]
            
            os.system(clear_command)
            print(self.build_cli_text(previous_history + new_history), flush=True)
            history = new_history[-self.history_maxlen:]
            if len(new_history) > self.history_maxlen:
                previous_history += new_history[:-self.history_maxlen]
            torch.cuda.empty_cache()


class ChatWebDemo(Chat):
    '''gradio实现的网页交互的demo'''
    def __init__(self, *args, max_length=4096, **generation_config):
        super().__init__(*args, **generation_config)
        import gradio as gr
        self.gr = gr
        self.max_length = max_length

    def reset_user_input(self):
        return self.gr.update(value='')

    @staticmethod
    def reset_state():
        return [], []

    def set_generation_config(self, max_length, top_p, temperature):
        '''根据web界面的参数修改生成参数'''
        self.generation_config['max_length'] = max_length
        self.generation_config['top_p'] = top_p
        self.generation_config['temperature'] = temperature

    def __stream_predict(self, input, chatbot, history):
        '''流式生成'''
        chatbot.append((input, ""))
        input_text = self.build_prompt(input, history)
        for response in self.model.stream_generate(input_text, **self.generation_config):
            response = self.process_response(response)
            chatbot[-1] = (input, response)
            new_history = history + [(input, response)]
            yield chatbot, new_history
        torch.cuda.empty_cache()  # 清理显存

    def __predict(self, input, chatbot, history):
        '''一次性生成'''
        chatbot.append((input, ""))
        input_text = self.build_prompt(input, history)
        response = self.model.generate(input_text, **self.generation_config)
        response = self.process_response(response)
        chatbot[-1] = (input, response)
        new_history = history + [(input, response)]
        torch.cuda.empty_cache()  # 清理显存
        return chatbot, new_history

    def run(self):
        with self.gr.Blocks() as demo:
            self.gr.HTML("""<h1 align="center">Chabot Web Demo</h1>""")

            chatbot = self.gr.Chatbot()
            with self.gr.Row():
                with self.gr.Column(scale=4):
                    with self.gr.Column(scale=12):
                        user_input = self.gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
                    with self.gr.Column(min_width=32, scale=1):
                        submitBtn = self.gr.Button("Submit", variant="primary")
                with self.gr.Column(scale=1):
                    emptyBtn = self.gr.Button("Clear History")
                    max_length = self.gr.Slider(0, self.max_length, value=self.max_length//2, step=1.0, label="Maximum length", interactive=True)
                    top_p = self.gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature = self.gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

            history = self.gr.State([])
            self.set_generation_config(max_length, top_p, temperature)
            if True:
                submitBtn.click(self.__stream_predict, [user_input, chatbot, history], [chatbot, history], show_progress=True)
            else:
                submitBtn.click(self.__predict, [user_input, chatbot, history], [chatbot, history], show_progress=True)

            submitBtn.click(self.reset_user_input, [], [user_input])
            emptyBtn.click(self.reset_state, outputs=[chatbot, history], show_progress=True)

        demo.queue().launch(share=True, inbrowser=True)


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
    role: str
    content: str


class DeltaMessage(BaseModel):
    role: Optional[str] = None
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


class ChatOpenaiApi(Chat):
    """OpenAi的api

    :param model_path: str, 模型所在的文件夹地址
    :param name: str, 模型名称
    :param generation_config: dict, 模型generate的参数设置
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    """
    def __init__(self, model_path, generation_config=None, name='default', route_api='/chat', route_models='/models') -> None:
        super().__init__(model_path, generation_config or dict())
        self.name = name
        self.role_user = 'user'
        self.role_assistant = 'assistant'
        self.role_system = 'system'
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
        log_warn_once('''The request post format should be 
            "messages": [{
                    "content": "你好",
                    "role": "user"
                },
                {
                    "content": "你好，我是法律大模型",
                    "role": "assistant"
                },
                {
                    "content": "基金从业可以购买股票吗",
                    "role": "user"
                }
            ],
            "model": "default",
            "stream": True
            ''')
        
    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        uvicorn.run(self.app, host=host, port=port, **kwargs)

    def build_prompt(self, messages):
        log_warn_once('Different Models have different prompt progress, You should custom define for each model.')
        if messages[-1].role != self.role_user:
            raise HTTPException(status_code=400, detail="Invalid request")
        query = messages[-1].content

        prev_messages = messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == self.role_system:
            query = prev_messages.pop(0).content + query

        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == self.role_user and prev_messages[i+1].role == self.role_assistant:
                    raise HTTPException(status_code=400, detail=f'Arg `messages` do not follow {self.role_user}, {self.role_assistant} format')
        messages = prev_messages + [{'role': i.role, 'content': i.content}]
        
        input_text = ''
        for msg in messages:
            input_text += msg['role'] + msg['content']
        return input_text

    async def list_models(self):
        model_card = ModelCard(id=self.name)
        return ModelList(data=[model_card])

    async def create_chat_completion(self, request: ChatCompletionRequest):
        input_text = self.build_prompt(request.messages)

        # 流式输出
        if request.stream:
            generate = self.predict(input_text, request.model)
            return EventSourceResponse(generate, media_type="text/event-stream")
        
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


class ChatOpenaiClient:
    '''调用openai接口的client
    '''
    def __init__(self, url) -> None:
        self.url = url
        if importlib.util.find_spec("sseclient") is None:
            raise ImportError('No module found, you may `pip install sseclient-py`')
        log_warn_once('''The body format should be 
            `reqBody = {
            "messages": [{
                    "content": "你好",
                    "role": "user"
                },
                {
                    "content": "你好，我是法律大模型",
                    "role": "assistant"
                },
                {
                    "content": "基金从业可以购买股票吗",
                    "role": "user"
                }
            ],
            "model": "default",
            "stream": True`
            ''')
   
    def post(self, body, verbose=1):
        reqHeaders = {'Accept': 'text/event-stream'}
        request = requests.post(self.url, stream=True, headers=reqHeaders, json=body)
        client = sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    if verbose:
                        print(data['content'], end="", flush=True)
                    yield data['content']
