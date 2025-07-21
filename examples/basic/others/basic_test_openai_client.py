'''测试openai接口'''
from bert4torch.snippets.openai_client import OpenaiClient, OpenaiClientSseclient, OpenaiClientAsync
from bert4torch.snippets import log_info
import time
import asyncio
import os

messages = [{"content": "你是谁", "role": "user"}]
messages_list = [
    [{"content": "你好", "role": "user"}],
    [{"content": "你是谁", "role": "user"}],
    [{"content": "1+1=？", "role": "user"}],
    ]

# TODO: 需要替换为自己的信息
base_url = 'https://api.siliconflow.cn/v1'
api_key = os.environ.get('API_KEY')
default_query = {}
default_headers = {}
model = 'Qwen/Qwen3-8B'

# ========================openai同步调用========================
client = OpenaiClient(
    base_url=base_url, 
    api_key=api_key, 
    default_headers=default_headers, 
    default_query=default_query
    )

log_info('ChatOpenaiClient.chat')
response = client.chat(messages, model)
print(response)

log_info('ChatOpenaiClient.stream_chat')
for response in client.stream_chat(messages, model):
    print(response, end='')
    time.sleep(0.05)
print()

# ========================openai异步调用========================
client = OpenaiClientAsync(
    base_url=base_url, 
    api_key=api_key, 
    default_headers=default_headers, 
    default_query=default_query
)


async def test_concurrent():
    log_info('ChatOpenaiClientAsync.chat')
    response = await client.chat(messages, model)
    print(response, '\n')

    log_info('ChatOpenaiClientAsync.stream_chat')
    async for res in client.stream_chat(messages, model):
        print(res, end='')
        await asyncio.sleep(0.05)
    print()

    log_info('ChatOpenaiClientAsync.batch_chat')
    await client.batch_chat(messages_list, model, verbose=True)

    log_info('ChatOpenaiClientAsync.stream_batch_chat')
    async for res in client.stream_batch_chat(messages_list, model, verbose=True):
        print(res)


asyncio.run(test_concurrent())
