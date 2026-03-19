''' 大模型聊天的pipeline调用
主要功能：
1. 命令行调用各个模型demo
2. 利用fastapi为大模型搭建openai格式的server和client调用
    Implements API for LLM in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
    Usage: python openai_api.py
    Visit http://localhost:8000/docs for documents.
3. web界面快速搭建demo(gradio+streamlit)

这里采用添加self.generation_config['states']['last_token']，是因为推理完成可能是因为到达max_length，未必是遇到了eos
'''

import torch
from typing import Union, Optional, List, Tuple, Literal, Dict, Type
from .big_modeling_llm import ChatBase, ChatCli, ChatWebGradio, ChatWebStreamlit, ChatOpenaiApi
from bert4torch.models.qwen2_vl.vision_process import MIN_PIXELS, MAX_PIXELS
from bert4torch.models.internvl.vision_process import fetch_image
from bert4torch.models.auto import AutoProcessor
from bert4torch.snippets import (
    log_warn_once, 
    get_config_path, 
    log_info, 
    log_info_once,
    log_warn, 
    log_error,
    colorful,
    cuda_empty_cache,
    is_fastapi_available, 
    is_pydantic_available, 
    is_streamlit_available,
    add_start_docstrings,
    is_transformers_available,
    load_image,
    create_registrar
)
import json
import copy
from PIL import Image
import inspect
import numpy as np
import os
import base64
from io import BytesIO
from argparse import REMAINDER, ArgumentParser


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
    'ChatVLBase',
    'MiniCPMV',
    'Qwen2VL',
    "Chat"
    ]

ImageType = Union[str, Image.Image, np.ndarray]
VedioType = Union[str, Image.Image, np.ndarray]

def trans_query_images_tolist(query, images):
    '''把query，images转为list'''
    def trans_images_to_Image(images:Union[ImageType, List[ImageType], List[List[ImageType]]]):
        '''把各种类型的images转化为Image.Image格式'''
        if isinstance(images, str) or isinstance(images, np.ndarray):
            images = load_image(images)
        elif isinstance(images, List) and all([isinstance(image, (str, Image.Image, np.ndarray)) for image in images]):
            images = [trans_images_to_Image(image) for image in images]
        elif isinstance(images, List) and all([isinstance(image, List) for image in images]):
            images = [trans_images_to_Image(image) for image in images]
        return images

    images = trans_images_to_Image(images)

    if isinstance(query, str):
        query = [query]
        if isinstance(images, Image.Image):
            # 提问单张图片
            images = [images]
        elif isinstance(images, List) and all([isinstance(image, Image.Image) for image in images]):
            # 同时提问多张图片
            images = [images]
        elif images is None:
            images = [images]
    elif isinstance(query, List) and all([isinstance(query, str) for query in query]):
        if isinstance(images, Image.Image):
            # 多次提问单张图片
            images = [images] * len(query)
        elif isinstance(images, List) and all([isinstance(image, Image.Image) for image in images]):
            # 各自提问单张图片
            pass
        elif isinstance(images, List) and all([isinstance(image, List) for image in images]) and \
            all([isinstance(i, Image.Image) for image in images for i in image]):
            # 各自同时提问多张图片
            pass
    return query, images

def trans_history_format2openai(history, image_key='image', vedio_key='vedio'):
    '''对history的格式转换为openai格式
    目前格式为：
    [
        {"role":"user", "images": [PIL.Image.Image, PIL.Image.Image], "content": "图片中描述了什么？"},
        {"role":"assistant", "content": "图片中描述了一只狗和一个少女在沙滩上玩耍的故事"},
    ]

    openai格式：image_url可替换为image
    [
        {"role":"user", "content": 
            [
                {"type": "image_url", "image_url": PIL.Image.Image},
                {"type": "text", "text": "图片中描述了什么？"}
            ]
        },
        {"role":"assistant", "content": "图片中描述了一只狗和一个少女在沙滩上玩耍的故事"},
    ]
    '''
    if not history:
        return []
    
    messages = []
    for hist in history:
        images = hist.get('images', [])
        content = [{"type": image_key, image_key: i} for i in images] if isinstance(images, list) \
            else [{"type": image_key, image_key: images}]
        content.append({'type': 'text', 'text': hist['content']})
        messages.append({'role': hist['role'], 'content': content})
    return messages

class ChatVLBase(ChatBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_tensorDict_from_build_prompt = True  # build_prompt返回的是字典
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path, trust_remote_code=True)

    @staticmethod
    def trans_history_format(history, format:Literal['openai', 'raw']='openai'):
        '''对history的格式转换，转换后可apply_chat_template
        目前格式为：
        [
            {"role":"user", "images": [PIL.Image.Image, PIL.Image.Image], "content": "图片中描述了什么？"},
            {"role":"assistant", "content": "图片中描述了一只狗和一个少女在沙滩上玩耍的故事"},
        ]

        openai格式：image_url可替换为image
        [
            {"role":"user", "content": 
                [
                    {"type": "image_url", "image_url": PIL.Image.Image},
                    {"type": "text", "text": "图片中描述了什么？"}
                ]
            },
            {"role":"assistant", "content": "图片中描述了一只狗和一个少女在沙滩上玩耍的故事"},
        ]
        '''
        return trans_history_format2openai(history)
    
    def chat(self, query:Union[str, List[str]], images:Union[ImageType, List[ImageType]]=None, vedios=None, 
             history:List[dict]=None, functions:List[dict]=None, return_history:bool=False,**kwargs) -> Union[str, List[str]]:
        '''chat模型使用, 配合对话模板使用'''
        history = history or []

        if isinstance(query, str) or self.return_tensorDict_from_build_prompt:
            # 单条输入，或在build_prompt阶段即组建了batch的tensor
            inputs:Dict[torch.Tensor] = self.build_prompt(query, images, vedios, history, functions, **kwargs)
            response = self.model.generate(**inputs, **self.generation_config)
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
            history_copy = [copy.deepcopy(history) for _ in query]
            images = [images] * len(query) if images is None or isinstance(images, (str, Image.Image, np.ndarray)) else images
            vedios = [vedios] * len(query) if vedios is None or isinstance(vedios, (str, Image.Image, np.ndarray)) else vedios
            inputs:List[Dict[torch.Tensor]] = [self.build_prompt(q, img, ved, hist, functions, **kwargs) for q, img, ved, hist in zip(query, images, vedios, history_copy)]
            response = self.model.generate(inputs, **self.generation_config)
            response = [self.process_response_history(r, history=hist) for r, hist in zip(response, history_copy)]
        else:
            raise TypeError(f'Args `query` type={type(query)} which is not supported')
        if return_history:
            return response, history
        else:
            return response

    def stream_chat(self, query:str, images:Union[ImageType, List[ImageType]]=None, 
                    vedios=None, history:List[dict]=None, functions:List[dict]=None, **kwargs):
        '''chat模型使用, 配合对话模板使用, 单条样本stream输出预测的结果'''
        history = history or []
        inputs = self.build_prompt(query, images, vedios, history, functions, **kwargs)
        for response in self.model.stream_generate(**inputs, **self.generation_config):
            yield self.process_response_history(response, history)
    
    def generate(self, *args, **kwargs):
        '''base模型使用'''
        return self.model.generate(*args, **self.generation_config, **kwargs)

    def stream_generate(self, *args, **kwargs):
        '''base模型使用, 单条样本stream输出预测的结果'''
        yield from self.model.stream_generate(*args, **self.generation_config, **kwargs)

    @staticmethod
    def update_history(history:List[Dict], query_list:List[str], image_list:Union[ImageType, List[ImageType]], raw_images=None):
        '''更新history'''
        for query, image in zip(query_list, image_list):
            history.append({'role': 'user', 'content': query})
            if image is None:
                continue
            elif isinstance(image, List):
                if all([i is not None for i in image]):
                    history[-1]['images'] = image
            elif isinstance(image, ImageType):
                history[-1]['images'] = image
            if raw_images is not None and isinstance(raw_images, str):  # 记录原始图片
                history[-1]['raw_images'] = raw_images
        return history


class ChatVLCli(ChatCli):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_str = kwargs.get('init_str', "输入内容进行对话，clear清空对话历史；stop终止程序；image输入图片路径或url，image为空表示不使用图片或使用上一次图片")

    def build_cli_text(self, history:List[dict]) -> str:
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for query_or_response in history:
            # 现在的dict格式，形如{'role': 'user', 'content': '你好啊'}
            if query_or_response['role'] == "user":
                prompt += f"\n\n{colorful('User：', color='green')}{query_or_response['content']}"
                if isinstance(query_or_response.get('raw_images', -1), str):
                    prompt += f"\n{colorful('Image：', color='green')}" + query_or_response['raw_images']
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
            
            images = input(f"{colorful('Image：', color='green')}")
            images = None if images.strip() == '' else images
            input_kwargs = self.build_prompt(query, images, None, history, functions)
            # history是human和assistant的聊天历史
            # 格式如[{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '有什么可以帮您的？'}]
            if stream:
                for response in self.model.stream_generate(**input_kwargs, **self.generation_config):
                    response = self.process_response_history(response, history)
                    os.system(clear_command)
                    print(self.build_cli_text(history), flush=True)
            else:
                response = self.model.generate(**input_kwargs, **self.generation_config)
                response = self.process_response_history(response, history)
                os.system(clear_command)
                print(self.build_cli_text(history), flush=True)
            cuda_empty_cache()


class ChatVLWebGradio(ChatWebGradio):
    '''需要添加一个图片的上传'''
    @staticmethod
    def get_image_vedio(chatbot):
        def _is_video_file(filename):
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
            return any(filename.lower().endswith(ext) for ext in video_extensions)
        
        input_image, input_vedio = None, None
        if chatbot and isinstance(chatbot[-1][0], tuple) and os.path.isfile(chatbot[-1][0][0]):
            if _is_video_file(chatbot[-1][0][0]):
                input_vedio = chatbot[-1][0][0]  # 视频
            else:
                input_image = chatbot[-1][0][0]  # 图片
        return input_image, input_vedio
            

    def _stream_predict(self, query, chatbot, history, max_length, top_p, temperature, repetition_penalty, system, functions):
        '''流式生成'''
        self.set_generation_config(max_length, top_p, temperature, repetition_penalty)
        input_image, input_vedio = self.get_image_vedio(chatbot)
        chatbot.append((query, ""))
        self._set_system_functions(system, functions)
        input_kwargs = self.build_prompt(query, input_image, input_vedio, history, functions)
        for response in self.model.stream_generate(**input_kwargs, **self.generation_config):
            response = self.process_response_history(response, history)
            if history[-1].get('raw_content'):
                response = history[-1]['raw_content']
            if history[-1].get('function_call'):
                response += f"\n\nFunction：{history[-1]['function_call']}"
            chatbot[-1] = (query, response)
            yield chatbot, history
        cuda_empty_cache()  # 清理显存

    def run(self, host:str=None, port:int=None, **launch_configs):

        def add_file(chatbot, file):
            chatbot = chatbot if chatbot is not None else []
            chatbot = chatbot + [((file.name,), None)]
            return chatbot

        with self.gr.Blocks() as demo:
            self.gr.HTML("""<h1 align="center">Chabot Gradio Demo</h1>""")
            with self.gr.Row():
                with self.gr.Column(scale=4):
                    chatbot = self.gr.Chatbot(height=500)
                    with self.gr.Column(scale=12):
                        query = self.gr.Textbox(show_label=False, placeholder="Input...", lines=10, max_lines=10) # .style(container=False)
                    with self.gr.Row():
                        addfile_btn = self.gr.UploadButton('📁 Upload', file_types=['image', 'video'])
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
            addfile_btn.upload(add_file, [chatbot, addfile_btn], [chatbot], show_progress=True)
            submitBtn.click(self._stream_predict, _input_tuple, [chatbot, history], show_progress=True)
            submitBtn.click(self.reset_user_input, [], [query])
            regen_btn.click(self.regenerate, _input_tuple, [chatbot, history], show_progress=True)
            emptyBtn.click(self.reset_state, outputs=[chatbot, history], show_progress=True)

        demo.queue().launch(server_name = launch_configs.pop('server_name', host), 
                            server_port = launch_configs.pop('server_port', port), 
                            **launch_configs)


class ChatVLWebStreamlit(ChatWebStreamlit):
    def run(self, debug:bool=False):
        def check_img_in_history(history, tgt_img):
            for message in history:
                for img in message.get('images', []):
                    if isinstance(img, list) and any([i == tgt_img for i in img]):
                        return True
                    elif img == tgt_img:
                        return True
            return False

        if "history" not in st.session_state:
            st.session_state.history = []
        if "states" not in st.session_state:
            st.session_state.states = None

        max_length = st.sidebar.slider("max_length", 0, self.max_length, self.max_length, step=1)
        top_p = st.sidebar.slider("top_p", 0.0, 1.0, self.generation_config.get('top_p', 1.0), step=0.01)
        temperature = st.sidebar.slider("temperature", 0.0, self.max_temperature, self.generation_config.get('temperature', 1.0), step=0.01)
        repetition_penalty = st.sidebar.slider("repetition_penalty", 0.0, self.max_repetition_penalty, self.generation_config.get('repetition_penalty', 1.0), step=0.1)
        buttonClean = st.sidebar.button("Clear history", key="clean")
        if buttonClean:
            st.session_state.history = []
            st.session_state.states = None
            cuda_empty_cache()
            st.rerun()
        
        # Select mode
        selected_mode = st.sidebar.selectbox("Select Mode", ["Text", "Images", "Video"])

        # Supported image file extensions
        image_type = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        for i, message in enumerate(st.session_state.history):
            role = message['role']
            if role not in {'user', 'assistant'}:
                continue
            if role == 'user':
                if message.get('images') is not None:
                    for img in message['images']:
                        if isinstance(img, list):
                            if not img:
                                continue
                            with st.chat_message(name=role, avatar=role):
                                for i in img:
                                    st.image(i, caption='User Uploaded Image', width=512, use_column_width=False)
                        else:
                            with st.chat_message(name=role, avatar=role):
                                st.image(img, caption='User Uploaded Image', width=512, use_column_width=False)
                if message.get('vedio') is not None:
                    with st.chat_message(name=role, avatar=role):
                        st.video(message['vedio'], format="video/mp4", loop=False, autoplay=False, muted=True) 
            with st.chat_message(name=role, avatar=role):
                st.markdown(message.get('raw_content', message['content']))
        
        images = []
        if selected_mode == "Images":
            # Multiple Images Mode
            uploaded_image_list = st.sidebar.file_uploader("Upload Images", key=2, type=image_type, accept_multiple_files=True)
            if uploaded_image_list is not None:
                for img in uploaded_image_list:
                    # st.image(img, caption='User Uploaded Image', width=512, use_column_width=False)
                    # 判断img是否在历史中
                    img_numpy = Image.open(img).convert('RGB')
                    if not check_img_in_history(st.session_state.history, img_numpy):
                        with st.chat_message(name='user', avatar='user'):
                            st.image(img, caption='User Uploaded Image', width=512, use_column_width=False)
                        images.append(img_numpy)

        # Supported video format suffixes
        video_type = ['.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v']
        videos = []
        # Tip: You can use the command `streamlit run ./web_demo_streamlit-minicpmv2_6.py --server.maxUploadSize 1024`
        # to adjust the maximum upload size to 1024MB or larger files.
        # The default 200MB limit of Streamlit's file_uploader component might be insufficient for video-based interactions.
        # Adjust the size based on your GPU memory usage.

        if selected_mode == "Video":
            # 单个视频模态
            uploaded_video = st.sidebar.file_uploader("Upload a single video file", key=3, type=video_type,
                                                    accept_multiple_files=False)
            if uploaded_video is not None:
                # st.video(uploaded_video, format="video/mp4", loop=False, autoplay=False, muted=True)
                with st.chat_message(name='user', avatar='user'):
                    st.video(uploaded_video, format="video/mp4", loop=False, autoplay=False, muted=True)
                videos.append(uploaded_video)

                uploaded_video_path = os.path.join(".\\uploads", uploaded_video.name)
                with open(uploaded_video_path, "wb") as vf:
                    vf.write(uploaded_video.getvalue())
                
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

        if system is not None and system.strip() != '':
            self.system = system
        
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

                if debug:
                    log_info(f'History before generate: {history}')
                input_kwargs = self.build_prompt(query, images, videos, history, functions)
                for response in self.model.stream_generate(**input_kwargs, **self.generation_config):
                    response = self.process_response_history(response, history)
                    message_placeholder.markdown(history[-1].get('raw_content', response))
                st.session_state.history = history
                if debug:
                    log_info(f'History after generate: {history}')
                st.session_state.states = self.generation_config.get('states')


class ChatVLOpenaiApi(ChatOpenaiApi):
    '''
    请求示例：
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这张图片是什么地方？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Either an url or base64
                        "url": "http://djclub.cdn.bcebos.com/uploads/images/pageimg/20230325/64-2303252115313.jpg"
                    }
                }
            ]
        }
    ]
    '''
    def prepare_build_prompt_args(self, request):
        content = request.messages[-1].content
        def get_query_images(content):
            if isinstance(content, str):
                query = content
                images = None
            else:  # dict
                query = [i for i in content if i['type']=='text'][0]['text']
                images = [Image.open(BytesIO(base64.b64decode(i['url']))).convert('RGB') for i in content if i['type']=='image_url']
            return query, images
        query, images = get_query_images(content)
        history = []
        for item in request.messages[:-1]:
            item_query, item_images = get_query_images(item.content)
            history.append({'role': item.role, 'content': item_query, 'images': item_images})
        input_kwargs = self.build_prompt(query, images, None, history, request.functions)
        return input_kwargs, history


# ==========================================================================================
# =========================              各个具体模型实现        ============================
# ==========================================================================================
VLM_MAPPING : Dict[str, Type[ChatVLBase]] = {}
register_vlm = create_registrar(VLM_MAPPING)


@register_vlm(name="minicpmv")
@register_vlm(name="minicpm_llama3_v")
class MiniCPMV(ChatVLBase):
    @staticmethod
    def trans_history_format(history):
        history_images = []
        history_copy = copy.deepcopy(history) if history else []
        for i, hist in enumerate(history_copy):
            role = hist["role"]
            content = hist["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            
            if 'images' not in hist:
                continue
            if isinstance(hist["images"], Image.Image):
                hist["images"] = [hist["images"]]
            hist["content"] = "(<image>./</image>)\n" * len(hist["images"]) + content
            history_images.extend(hist["images"])
        return history_copy, history_images

    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image]]=None, 
            vedios=None, history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs):
        '''
        history: [
                    {'role': 'user', 'content': '图片中描述的是什么', 'images': [PIL.Image.Image]},
                    {'role': 'assistant', 'content': '该图片中描述了一个小男孩在踢足球'},
                 ]

        |   queries   |        images      |     comment      |
        |   -------   |      --------      |    ---------     |
        |     str     |        Image       |    提问单张图片   |
        |     str     |     List[Image]    |  同时提问多张图片  |
        |  List[str]  |        Image       |  多次提问单张图片  |
        |  List[str]  |     List[Image]    |  各自提问单张图片  |
        |  List[str]  |  List[List[Image]] |各自同时提问多张图片|
        '''
        query_list, image_list = trans_query_images_tolist(queries, images)

        assert len(query_list) == len(image_list), "The batch dim of query and images should be the same."        
        assert self.model.config.query_num == self.processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.patch_size == self.processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.use_image_id == self.processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.slice_config.max_slice_nums == self.processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.slice_mode == self.processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        # 处理history
        history_messages, history_images = self.trans_history_format(history)

        prompts_lists = []
        input_images_lists = []
        for q, image in zip(query_list, image_list):
            copy_msgs = copy.deepcopy(history_messages) if history else []
            if image is None:
                image = []
            elif isinstance(image, Image.Image):
                image = [image]
            content = "(<image>./</image>)\n"*len(image) + q
            copy_msgs.append({'role': 'user', 'content': content})

            if kwargs.get('system_prompt'):
                sys_msg = {'role': 'system', 'content': kwargs.get('system_prompt')}
                copy_msgs = [sys_msg] + copy_msgs        

            prompts_lists.append(self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(history_images + image)
        
        if 'max_slice_nums' in inspect.signature(self.processor).parameters:
            # MiniCPM-V-2_6
            inputs = self.processor(
                prompts_lists, 
                input_images_lists, 
                max_slice_nums=kwargs.get('max_slice_nums'),
                use_image_id=kwargs.get('use_image_id'),
                return_tensors="pt", 
                max_length=kwargs.get('max_inp_length'),
            ).to(self.device)
        else:
            # MiniCPM-Llama3-V-2_5, 仅接受单张照片预测
            if len(prompts_lists) > 1:
                raise ValueError('`MiniCPM-Llama3-V-2_5` not support batch inference.')
            inputs = self.processor(
                prompts_lists[0], 
                input_images_lists[0], 
                return_tensors="pt", 
                max_length=kwargs.get('max_inp_length'),
            ).to(self.device)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'], dtype=bool)

        inputs.pop("image_sizes")
        history = self.update_history(history, query_list, image_list, raw_images=images)
        return inputs


@register_vlm(name='qwen2_vl')
@register_vlm(name='qwen2_5_vl')
@register_vlm(name='qwen3_vl')
class Qwen2VL(ChatVLBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_pixels = kwargs.get('min_pixels', MIN_PIXELS)
        self.max_pixels = kwargs.get('max_pixels', MAX_PIXELS)
        log_warn_once('Please set `max_pixels` smaller when CUDA out_of_memory eccured')
    
    def trans_history_format(self, history):
        messages = super().trans_history_format(history)
        for msg in messages:
            for c in msg['content']:
                if c['type'] == 'image':
                    c.update({"min_pixels":self.min_pixels, "max_pixels": self.max_pixels})
        return messages
    
    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]]=None, 
            vedios=None,
            history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs
        ):
        if self.system is not None and not history:
            history.append({'role': 'system', 'content': self.system})
        query_list, image_list = trans_query_images_tolist(queries, images)
        history_messages = self.trans_history_format(history)

        all_messages = []
        for query, image in zip(query_list, image_list):
            messages = copy.deepcopy(history_messages) + [{"role": "user", "content": [{"type": "text", "text": query}]}]
            if image is None:
                pass
            elif isinstance(image, list):
                messages[-1]['content'] = [{"type": "image", "image": i, "min_pixels":self.min_pixels, 
                                            "max_pixels": self.max_pixels} for i in image] + messages[-1]['content']
            else:
                messages[-1]['content'] = [{"type": "image", "image": image, "min_pixels":self.min_pixels, 
                                            "max_pixels": self.max_pixels}] + messages[-1]['content']
            all_messages.append(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        history = self.update_history(history, query_list, image_list, raw_images=images)
        return inputs


@register_vlm(name="glm_ocr")
@register_vlm(name="mllama")
@register_vlm(name="paddleocr_vl")
class Mllama(ChatVLBase):
    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]]=None, 
            vedios=None,
            history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs
        ):
        if self.system is not None and not history:
            history.append({'role': 'system', 'content': self.system})
        query_list, image_list = trans_query_images_tolist(queries, images)
        history_messages = self.trans_history_format(history)

        all_messages = []
        for query, image in zip(query_list, image_list):
            messages = copy.deepcopy(history_messages) + [{"role": "user", "content": [{"type": "text", "text": query}]}]
            if image is None:
                pass
            elif isinstance(image, list):
                messages[-1]['content'] = [{"type": "image", "image": i} for i in image] + messages[-1]['content']
            else:
                messages[-1]['content'] = [{"type": "image", "image": image}] + messages[-1]['content']
            all_messages.append(messages)

        input_text = self.processor.apply_chat_template(all_messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        history = self.update_history(history, query_list, image_list, raw_images=images)
        return inputs
    

@register_vlm(name="glm4v")
class GLM4V(ChatVLBase):
    @staticmethod
    def trans_history_format(history):
        history_messages = []
        for msg in history or []:
            if 'images' in msg:
                history_messages.append({'image':msg['images'], **{k:v for k,v in msg.items() if k != 'images'}})
            else:
                history_messages.append(msg)
        return history_messages

    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]]=None, 
            vedios=None,
            history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs
        ):
        # 整个message中只能只能允许一张图片
        if self.system is not None and not history:
            history.append({'role': 'system', 'content': self.system})
        query_list, image_list = trans_query_images_tolist(queries, images)
        history_messages = self.trans_history_format(history)

        all_messages = []
        for query, image in zip(query_list, image_list):
            messages = copy.deepcopy(history_messages) + [{"role": "user", "content": query}]
            if image is not None:
                messages[-1]['image'] = image
            # message中如果有多张图片，只保留最后一张
            image_count = 0
            for item in messages[::-1]:
                if 'image' not in item:
                    continue
                elif isinstance(item['image'], list):
                    if image_count > 0:
                        image_count += len(item['image'])
                        item.pop('image')
                    else:
                        image_count += len(item['image'])
                        item['image'] = item['image'][-1]
                else:
                    if image_count > 0:
                        item.pop('image')
                    image_count += 1
            if image_count > 1:
                log_warn(f'glm4v only can process one image in one turn chat, but got len(image)={image_count}, use last image instead.')                
            all_messages.append(messages)

        inputs: dict = self.tokenizer.apply_chat_template(all_messages, add_generation_prompt=True, tokenize=True, 
            return_tensors="pt", return_dict=True).to(self.device)

        history = self.update_history(history, query_list, image_list, raw_images=images)
        return inputs


@register_vlm(name="internvl2_5")
class InternVL(ChatVLBase):
    def __init__(self, *args, max_num:int=12, separate_or_conbined_images:Literal['separate', 'conbined']='separate', **kwargs):
        super().__init__(*args, **kwargs)
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))
        self.max_num = max_num
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.separate_or_conbined_images = separate_or_conbined_images

    def trans_image2pixel_values(self, images, query):
        images = [images] if not isinstance(images, (tuple,list)) else images
        pixel_value_list = [fetch_image(image, max_num=self.max_num).to(torch.bfloat16).to(self.device) for image in images]
        pixel_values = torch.cat(pixel_value_list, dim=0)
        if self.separate_or_conbined_images == 'separate':
            content = '<image>\n' * len(images) + query
            num_patches_list = [pixel.shape[0] for pixel in pixel_value_list]
        else:
            content = '<image>\n' + query
            num_patches_list = [pixel_values.shape[0]]
        return content, pixel_values, num_patches_list

    def trans_history_format(self, history):
        history_messages = []
        history_pixel_values = []
        history_num_patches_list = []
        for hist in history:
            images = hist.get('images', [])
            if images:
                # 有图片
                content, pixel_values, num_patches_list = self.trans_image2pixel_values(images, hist['content'])
                history_pixel_values.append(pixel_values)
                history_num_patches_list.extend(num_patches_list)
            else:
                content = hist['content']
            history_messages.append({'role': hist['role'], 'content': content})        
        return history_messages, history_pixel_values, history_num_patches_list
    
    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]]=None, 
            vedios=None,
            history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs
        ):
        query_list, image_list = trans_query_images_tolist(queries, images)
        history_messages, history_pixel_values_list, history_num_patches_list = self.trans_history_format(history)
        
        query_input = []
        pixel_values_list = []
        for query, image in zip(query_list, image_list):
            if image is not None:
                query, pixel_values, num_patches_list = self.trans_image2pixel_values(image, query)
            else:
                pixel_values = None
                num_patches_list = []

            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
            if history_pixel_values_list:
                pixel_values = torch.cat(history_pixel_values_list + ([pixel_values] if pixel_values else []), dim=0)

            template = self.build_template(system_message=self.system)
            for hist in history_messages:
                if hist['role'] == 'user':
                    template.append_message(template.role_user, hist['content'])
                elif hist['role'] == 'assistant':
                    template.append_message(template.role_assistant, hist['content'])
            template.append_message(template.role_user, query)
            template.append_message(template.role_assistant, None)
            query = template.get_prompt()

            for num_patches in history_num_patches_list + num_patches_list:
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + self.IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            query_input.append(query)
            pixel_values_list.append(pixel_values)

        history = self.update_history(history, query_list, image_list, raw_images=images)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(query_input, return_tensors='pt', padding=True).to(self.device)
        if all([i is not None for i in pixel_values_list]):
            model_inputs['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        else:
            model_inputs['pixel_values'] = None
        return model_inputs
