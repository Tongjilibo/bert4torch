'''大模型的chat pipeline
   1) 命令行聊天
   2) gradio和streamlit的网页demo
   3) 发布类openai的api接口
'''
from .base import Chat, ChatCli, ChatWebGradio, ChatWebStreamlit
from .openai_api import ChatOpenaiApi, ChatOpenaiClient, ChatOpenaiClientSseclient
from .llm import *