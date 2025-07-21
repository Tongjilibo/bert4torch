'''调用大模型接口的封装
'''
from typing import List, Dict, Union, Optional, Literal
import asyncio
from openai import OpenAI, AsyncOpenAI
from bert4torch.snippets import is_sseclient_available, log_info, log_error, green, red
import traceback
import requests
import json
import random
import time
import inspect


class OpenaiClient:
    '''使用openai来调用，单线程，单条信息
    
    Examples:
    ```python
    >>> messages = [
    ...         {"content": "你好", "role": "user"},
    ...         {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
    ...         {"content": "你可以做什么？", "role": "user"}
    ...         ]
    >>> client = OpenaiClient('http://127.0.0.1:8000')

    >>> # 流式
    >>> for token in client.stream_chat(messages, model='Qwen3-0.6B'):
    ...     print(token, end='', flush=True)

    >>> # 非流式
    >>> print(client.chat(messages, model='Qwen3-0.6B'))
    ```
    '''
    def __init__(self, base_url:Union[str, List[str]], api_key:Union[str, List[str]]=None, 
                 client_select_mode:Literal['random', 'sort']='random', 
                 return_content:Literal['response', 'content']='content', **kwargs) -> None:
        self.client_select_mode = client_select_mode
        self.client_id = 0
        self.return_content = return_content

        if isinstance(base_url, str):
            self.clients = OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        elif isinstance(base_url, list):
            self.clients = [OpenAI(base_url=url, api_key=key, **kwargs) for url, key in zip(base_url, api_key)]
        else:
            raise TypeError('Args `base_url` and `api_key` only support str,List[str] format')

    def get_client(self):
        '''获取client'''
        if isinstance(self.clients, list):
            if self.client_select_mode == 'random':
                random.seed(time.time())
                client = random.choice(self.clients)
            elif self.client_select_mode == 'sort':
                client = self.clients[self.client_id]
                self.client_id = (self.client_id + 1) % len(self.clients)
            else:
                raise ValueError('Args `client_select_mode` only support random/sort')
            return client
        else:
            return self.clients

    def get_content(self, response_or_chunk):
        '''返回自己的内容'''
        if self.return_content == 'content':
            # 返回文字
            if hasattr(response_or_chunk.choices[0], 'delta'):
                # 流式
                reasoning_content = getattr(response_or_chunk.choices[0].delta, 'reasoning_content', '') or ''
                content = getattr(response_or_chunk.choices[0].delta, 'content', '') or ''
                return reasoning_content + content
            elif hasattr(response_or_chunk.choices[0], 'message'):
                # 非流式
                reasoning_content = getattr(response_or_chunk.choices[0].message, 'reasoning_content', '') or ''
                content = getattr(response_or_chunk.choices[0].message, 'content', '') or ''
                return reasoning_content + content
        else:
            # 想要使用functions或者tools，外部需要自行解析
            return response_or_chunk

    @staticmethod
    def get_create_kwargs(kwargs):
        '''只传入create的参数'''
        return {k: v for k, v in kwargs.items() if k in inspect.signature(OpenAI.chat.completions.create).parameters}

    def stream_chat(self, messages:List[Dict], model:str, max_tokens:int=None, temperature:float=None, top_p:float=None, **kwargs):
        '''流式返回'''
        response = self.get_client().chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **self.get_create_kwargs(kwargs)
            )

        for chunk in response:
            content = self.get_content(chunk)
            if content is not None:
                yield content

    def chat(self, messages:List[Dict], model:str, max_tokens:int=None, temperature:float=None, top_p:float=None, **kwargs):
        '''一次性返回'''
        response = self.get_client().chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **self.get_create_kwargs(kwargs)
            )
        content = self.get_content(response)
        return content
    
    def stream_chat_cli(self, *args, **kwargs):
        '''流式返回并在命令行打印'''
        for token in self.stream_chat(*args, **kwargs):
            print(token, end='', flush=True)


class OpenaiClientAsync(OpenaiClient):
    '''使用openai来异步调用
    
    Examples:
    ```python
    >>> messages = [
    ...         {"content": "你好", "role": "user"},
    ...         {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
    ...         {"content": "你可以做什么？", "role": "user"}
    ...         ]
    >>> client = OpenaiClient('http://127.0.0.1:8000', api_key='EMPTY')

    >>> # 流式
    >>> for token in client.stream_chat(messages, model='Qwen3-0.6B'):
    ...     print(token, end='', flush=True)

    >>> # 非流式
    >>> print(client.chat(messages, model='Qwen3-0.6B'))
    ```
    '''
    def __init__(self, base_url:Union[str, List[str]], api_key:Union[str, List[str]]=None, 
                 client_select_mode:Literal['random', 'sort']='random', 
                 return_content:Literal['response', 'content']='content', **kwargs) -> None:
        self.client_select_mode = client_select_mode
        self.client_id = 0
        self.return_content = return_content

        if isinstance(base_url, str):
            self.clients = AsyncOpenAI(base_url=base_url, api_key=api_key, **kwargs)
        elif isinstance(base_url, list):
            self.clients = [AsyncOpenAI(base_url=url, api_key=key, **kwargs) for url, key in zip(base_url, api_key)]
        else:
            raise TypeError('Args `base_url` and `api_key` only support str,List[str] format')
        
    async def stream_chat(
        self, 
        messages:List[Dict], 
        model:str, 
        max_tokens:int=None, 
        temperature:float=None, 
        top_p:float=None, 
        **kwargs
    ):
        '''流式返回，单个请求'''
        response = await self.get_client().chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **self.get_create_kwargs(kwargs)
            )

        async for chunk in response:
            content = self.get_content(chunk)
            if content is not None:
                yield content

    async def chat(
        self, 
        messages:List[Dict], 
        model:str, 
        max_tokens:int=None, 
        temperature:float=None, 
        top_p:float=None, 
        **kwargs
    ):
        '''一次性返回，单个请求'''
        response = await self.get_client().chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **self.get_create_kwargs(kwargs)
            )
        content = self.get_content(response)
        return content
    
    async def stream_chat_cli(self, *args, **kwargs):
        async for token in self.stream_chat(*args, **kwargs):
            print(token, end='', flush=True)

    async def batch_chat(
        self,
        messages_list:List[List[Dict]], 
        model:str, 
        max_tokens:int=1024, 
        temperature:float=None, 
        top_p:float=None, 
        batch_size:Optional[int]=None, 
        workers:int=5, 
        verbose:bool=False,
        **kwargs
    ):
        """并发调用, 多个messages并发调用，分batch模式和限制并发模式，所有都请求完成后一起返回，默认使用限制并发模式"""
        async def generate_text(client, messages, max_token, semaphore=None):
            async def _create_completion():
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_token,
                    temperature=temperature,
                    top_p=top_p,
                    **self.get_create_kwargs(kwargs)
                )
                return self.get_content(response)
            
            if semaphore:
                async with semaphore:
                    return await _create_completion()
            else:
                return await _create_completion()
            
        assert (batch_size is not None) ^ (workers is not None), 'Exactly one of `batch_size` or `workers` must be specified'
        assert isinstance(messages_list, list), f'Args `messages_list` must be a list, not {type(messages_list)}'
        assert isinstance(messages_list[0], list), f'Args `messages_list` must be a List[List], not List[{type(messages_list[0])}]'
        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(messages_list)
        assert len(max_tokens) == len(messages_list)

        results = []
        if batch_size is not None:
            # 批量模式
            for i in range(0, len(messages_list), batch_size):
                batch_messages = messages_list[i:i+batch_size]
                batch_max_tokens = max_tokens[i:i+batch_size]
                if not batch_messages:
                    continue
                tasks = [generate_text(self.get_client(), m, t) for m, t in zip(batch_messages, batch_max_tokens)]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        else:
            # 并发限制模式
            workers = min(workers, len(messages_list))
            semaphore = asyncio.Semaphore(workers)
            tasks = [generate_text(self.get_client(), m, t, semaphore) for m, t in zip(messages_list, max_tokens)]
            results = await asyncio.gather(*tasks)
        
        if verbose: 
            for messages, result in zip(messages_list, results):
                print(f"{green('Prompt')}: {messages}")
                print(f"{green('Response')}: {result}\n")
        return results

    async def stream_batch_chat(
        self,
        messages_list:List[List[Dict]], 
        model_name:str, 
        workers:int=5, 
        max_tokens:Union[int, List[int]]=1024, 
        temperature:float=None, 
        top_p:float=None, 
        verbose:bool=False,
        **kwargs
    ):
        '''调用OpenAI API, 多个messages并发调用，generator先完成的先返回'''
        async def generate_text(client, messages, max_token, semaphore):
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_token,
                    temperature=temperature,
                    top_p=top_p,
                    **self.get_create_kwargs(kwargs)
                )
                return self.get_content(response)

        assert isinstance(messages_list, list), f'Args `messages_list` must be a list, not {type(messages_list)}'
        assert isinstance(messages_list[0], list), f'Args `messages_list` must be a List[List], not List[{type(messages_list[0])}]'
        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(messages_list)
        assert len(max_tokens) == len(messages_list)

        workers = min(workers, len(messages_list))
        semaphore = asyncio.Semaphore(workers)

        # 创建所有任务
        tasks = [generate_text(self.get_client(), m, t, semaphore) for m, t in zip(messages_list, max_tokens)]
        
        try:
            # 使用as_completed实现完成一个返回一个
            success, fail, total = 0, 0, len(tasks)
            for finished_task in asyncio.as_completed(tasks):
                try:
                    result = await finished_task
                    # 注意：这里无法直接获取对应的prompt，因为任务完成顺序不确定
                    success += 1
                    if verbose:
                        print(f"{green('Sucess')}: success={success}, fail={fail}, total={total}")
                        print(f"{green('Response')}: {result}\n")
                    yield result   # 使用生成器模式返回结果
                except Exception as e:
                    fail += 1
                    if verbose:
                        print(f"{red('Fail')}: success={success}, fail={fail}, total={total}; {str(e)}")
                    yield None
        except:
            log_error(traceback.format_exc())


class OpenaiClientSseclient:
    '''调用openai接口的client, 流式请求

    Examples:
    ```python
    >>> # 注意事项：部分调用时候有额外参数传入，如下：
    >>> client = OpenaiClientSseclient(url='https://chatpet.openai.azure.com/openai/deployments/chatGPT-turbo16K/chat/completions', 
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

    >>> client = OpenaiClientSseclient('http://127.0.0.1:8000')
    >>> # 测试打印
    >>> client.stream_chat_cli(body)

    >>> # 流式
    >>> for token in client.stream_chat(body):
    ...     print(token, end='', flush=True)
    ```
    '''
    def __init__(self, base_url:str, api_key:str=None, header:dict=None, params:dict=None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.header = header
        self.params = params

        if is_sseclient_available():
            import sseclient
        else:
            raise ModuleNotFoundError('No module found, you may `pip install sseclient-py`')
        
        self.sseclient = sseclient
   
    def stream_chat(self, messages:List[Dict], model:str, max_tokens:int=None, temperature:float=None, top_p:float=None, **kwargs):
        '''接口调用'''
        reqHeaders = {'Accept': 'text/event-stream'}
        if self.api_key is not None:
            reqHeaders["Authorization"] = f"Bearer {self.api_key}"

        if self.header is not None:
            reqHeaders.update(self.header)
        
        body = {'messages': messages, 'model': model, 'stream': True, 'max_tokens': max_tokens, 
                'temperature': temperature, 'top_p': top_p}
        request = requests.post(self.base_url, stream=True, headers=reqHeaders, json=body, params=self.params, **kwargs)
        client = self.sseclient.SSEClient(request)
        for event in client.events():
            if event.data != '[DONE]':
                data = json.loads(event.data)['choices'][0]['delta']
                if 'content' in data:
                    yield data['content']

    def stream_chat_cli(self, messages:List[Dict], **kwargs):
        '''简单测试在命令行打印'''
        for token in self.stream_chat(messages, **kwargs):
            print(token, end='', flush=True)