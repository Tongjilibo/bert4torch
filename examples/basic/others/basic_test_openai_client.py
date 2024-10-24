'''测试openai接口'''
from bert4torch.pipelines import ChatOpenaiClient, ChatOpenaiClientSseclient, ChatOpenaiClientAsync
from bert4torch.snippets import log_info
import time
import asyncio

messages = [{"content": "你是谁", "role": "user"}]
messages_list = [
    [{"content": "你好", "role": "user"}],
    [{"content": "你是谁", "role": "user"}],
    [{"content": "1+1=？", "role": "user"}],
    [{"content": "查询天气", "role": "user"}],
    [{"content": "感谢你", "role": "user"}],
    ]

# TODO: 需要替换为自己的信息
base_url = ''
api_key = ''
default_query = {}
default_headers = {}
model = ''

# openai同步调用
client = ChatOpenaiClient(base_url=base_url, 
                          api_key=api_key, 
                          default_headers=default_headers, 
                          default_query=default_query)

log_info('ChatOpenaiClient.chat')
response = client.chat(messages, model)
print(response)

log_info('ChatOpenaiClient.stream_chat')
for response in client.stream_chat(messages, model):
    print(response, end='')
    time.sleep(0.05)

# openai异步调用
client = ChatOpenaiClientAsync(
    base_url=base_url, 
    api_key=api_key, 
    default_headers=default_headers, 
    default_query=default_query
)
async def chat():
    response = await client.chat(messages, model)
    print(response)

async def stream_chat():
    async for res in client.stream_chat(messages, model):
        print(res, end='')
        await asyncio.sleep(0.05)
    print()

async def test_concurrent():
    log_info('ChatOpenaiClientAsync.chat')
    await chat()

    log_info('ChatOpenaiClientAsync.stream_chat')
    await stream_chat()

    log_info('ChatOpenaiClientAsync.concurrent_chat')
    await client.concurrent_chat(messages_list, model, verbose=True)

    log_info('ChatOpenaiClientAsync.batch_chat')
    await client.batch_chat(messages_list, model, verbose=True)

asyncio.run(test_concurrent())
