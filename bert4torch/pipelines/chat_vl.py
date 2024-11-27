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

import torch
from typing import Union, Optional, List, Tuple, Literal, Dict
from bert4torch.pipelines.chat import ChatBase, ChatWebGradio, ChatWebStreamlit, ChatOpenaiApi, CHAT_START_DOCSTRING, OPENAI_START_DOCSTRING
from bert4torch.pipelines.chat import ChatCompletionRequest, ChatCompletionResponseChoice, ChatMessage, ChatCompletionResponse, ChatCompletionResponseStreamChoice, DeltaMessage
from bert4torch.models.qwen2_vl import process_vision_info
from bert4torch.models.qwen2_vl.vision_process import MIN_PIXELS, MAX_PIXELS
from bert4torch.snippets import (
    log_warn_once, 
    get_config_path, 
    log_info, 
    log_info_once,
    log_warn, 
    log_error,
    cuda_empty_cache,
    is_fastapi_available, 
    is_pydantic_available, 
    is_sseclient_available, 
    is_streamlit_available,
    is_package_available,
    has_chinese_char,
    add_start_docstrings,
    JsonConfig,
    is_transformers_available,
    inference_mode,
    sequence_padding
)
import time
import json
import copy
from PIL import Image
import inspect
import numpy as np
import os
import base64
from io import BytesIO


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

if is_transformers_available():
    from transformers import AutoProcessor

__all__ = [
    'ChatVBase',
    'MiniCPMV',
    "ChatV"
    ]

ImageType = Union[str, Image.Image, np.ndarray]
VedioType = Union[str, Image.Image, np.ndarray]

def trans_query_images_tolist(query, images):
    '''把query，images转为list'''
    def trans_images_to_Image(images:Union[ImageType, List[ImageType], List[List[ImageType]]]):
        '''把各种类型的images转化为Image.Image格式'''
        if isinstance(images, str):
            images = Image.open(images).convert('RGB')
        elif isinstance(images, np.ndarray):
            images = Image.fromarray(images)
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

class ChatVBase(ChatBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_tensor_from_build_prompt = True
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def generate(self, *args, **kwargs):
        '''base模型使用'''
        return self.model.generate(*args, **kwargs)

    def build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @inference_mode()
    def chat(self, query:Union[str, List[str]], images:Union[ImageType, List[ImageType]]=None, 
             vedios=None, history:List[dict]=None, functions:List[dict]=None, **kwargs) -> Union[str, List[str]]:
        '''chat模型使用, 配合对话模板使用'''
        history = history or []

        if isinstance(query, str) or self.return_tensor_from_build_prompt:
            # 单条输入，或在build_prompt阶段即组建了batch的tensor
            inputs:Dict[torch.Tensor] = self.build_prompt(query, images, vedios, history, functions, **kwargs)
            response = self.model.generate(**inputs, **self.generation_config)
            if isinstance(response, str):
                # 生成单条输出
                return self.process_response_history(response, history=history)
            elif isinstance(response, list):
                # 为单条query生成多条response
                return [self.process_response_history(resp, history=copy.deepcopy(history)) for resp in response]
            else:
                raise TypeError(f'`response` type={type(response)} which is not supported')
            
        elif isinstance(query, list):
            # 多条输入
            history_copy = [copy.deepcopy(history) for _ in query]
            images = [images] * len(query) if images is None or isinstance(images, (str, Image.Image, np.ndarray)) else images
            vedios = [vedios] * len(query) if vedios is None or isinstance(vedios, (str, Image.Image, np.ndarray)) else vedios
            inputs:List[Dict[torch.Tensor]] = [self.build_prompt(q, img, ved, hist, functions, **kwargs) for q, img, ved, hist in zip(query, images, vedios, history_copy)]
            response = self.model.generate(inputs, **self.generation_config)
            return [self.process_response_history(r, history=hist) for r, hist in zip(response, history_copy)]
        else:
            raise TypeError(f'Args `query` type={type(query)} which is not supported')


class ChatVWebGradio(ChatWebGradio):
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


class ChatVWebStreamlit(ChatWebStreamlit):
    def run(self, debug:bool=False):
        def check_img_in_history(history, tgt_img):
            for i, message in enumerate(history):
                if message.get('images') is not None:
                    for img in message['images']:
                        if isinstance(img, list):
                            for i in img:
                                if i == tgt_img:
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


class ChatVOpenaiApi(ChatOpenaiApi):
    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.model != self.name:
            raise HTTPException(status_code=404, detail=f"Invalid model: request.model:{request.model} != self.name:{self.name}")
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

        content = request.messages[-1].content
        def get_query_images(content):
            if isinstance(content, str):
                query = content
                images = None
            else:  # dict
                query = [i for i in content if i['type']=='text'][0]['content']
                images = [Image.open(BytesIO(base64.b64decode(i['url']))).convert('RGB') for i in content if i['type']=='image_url']
            return query, images
        query, images = get_query_images(content)
        history = []
        for item in request.messages[:-1]:
            item_query, item_images = get_query_images(item.content)
            history.append({'role': item.role, 'content': item_query, 'images': item_images})
        input_kwargs = self.build_prompt(query, images, None, history, request.functions)
        
        if self.offload_when_nocall is None:
            self.model = self.build_model()
        else:
            with self.lock:
                self.model = self.build_model()
            self.last_callapi_timestamp = time.time()

        # 流式输出
        if request.stream:
            generate = self.predict(input_kwargs, request.model, history)
            return self.EventSourceResponse(generate, media_type="text/event-stream")
        
        # 非流式输出
        else:
            response = self.model.generate(**input_kwargs, **self.generation_config)
            response = self.process_response_history(response, history)
            function_call = history[-1].get('function_call', None)
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role=self.role_assistant, content=response, function_call=function_call),
                finish_reason= "function_call" if function_call is not None else "stop"
            )

            return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    async def predict(self, input_kwargs: dict, model_id: str, history:list):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=self.role_assistant),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        current_length = 0

        for new_response in self.model.stream_generate(**input_kwargs, **self.generation_config):
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


class MiniCPMV(ChatVBase):
    def build_prompt(
            self,
            queries: Union[str, list], 
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
        queries, images = trans_query_images_tolist(queries, images)

        assert len(queries) == len(images), "The batch dim of query and images should be the same."        
        assert self.model.config.query_num == self.processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.patch_size == self.processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.use_image_id == self.processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.slice_config.max_slice_nums == self.processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.slice_mode == self.processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        # 处理history
        history_images = []
        history_copy = copy.deepcopy(history) or []
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
            hist["content"] = "(<image>./</image>)\n"*len(hist["images"]) + content
            history_images.extend(hist["images"])

        prompts_lists = []
        input_images_lists = []
        for q, image in zip(queries, images):
            copy_msgs = copy.deepcopy(history_copy) or []
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
        history.append({'role': 'user', 'content': queries[0]})
        if all([i is not None for i in images]):
            history[-1]['images'] = images
        return inputs


class Qwen2VL(ChatVBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_pixels = kwargs.get('min_pixels', MIN_PIXELS)
        self.max_pixels = kwargs.get('max_pixels', MAX_PIXELS)
        log_warn_once('Please set `max_pixels` smaller when CUDA out_of_memory eccured')

    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]]=None, 
            vedios=None,
            history: List[Dict]=None, 
            functions:List[dict]=None,
            **kwargs
        ):
        if hasattr(self, 'system') and not history:
            history.append({'role': 'system', 'content': self.system})
        queries, images = trans_query_images_tolist(queries, images)

        all_messages = []
        for query, image in zip(queries, images):
            messages = (copy.deepcopy(history) or [] or []) + [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}]
                }]
            if image is None:
                pass
            elif isinstance(image, list):
                messages[-1]['content'] = [{"type": "image", "image": i, "min_pixels":self.min_pixels, 
                                            "max_pixels": self.max_pixels} for i in image] + messages[-1]['content']
            else:
                messages[-1]['content'] = [{"type": "image", "image": image, "min_pixels":self.min_pixels, 
                                            "max_pixels": self.max_pixels}] + messages[-1]['content']
            all_messages.append(messages)

        text = self.processor.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(all_messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        history.append({'role': 'user', 'content': queries[0]})
        if all([i is not None for i in images]):
            history[-1]['images'] = images
        return inputs


# ==========================================================================================
# =======================                统一Chat入口             ==========================
# ==========================================================================================
MAPPING = {
    'minicpmv': MiniCPMV,
    'qwen2_vl': Qwen2VL
}

class ChatV:
    """
    部署类似OpenAi的api server端

    ### 基础参数
    :param checkpoint_path: str, 模型所在的文件夹地址
    :param precision: bool, 精度, 'double', 'float', 'half', 'float16', 'bfloat16'
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quantization_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'max_length':2048, 'default_rtype':'logits', 'use_states':True}
        - bos_token_id: int, 解码使用的起始token_id, 不同预训练模型设置可能不一样
        - eos_token_id: int/tuple/list, 解码使用的结束token_id, 不同预训练模型设置可能不一样, 默认给的-1(真实场景中不存在, 表示输出到max_length)
        - max_new_tokens: int, 最大解码长度
        - min_new_tokens: int, 最小解码长度, 默认为1
        - max_length: int, 最大文本长度
        - pad_token_id: int, pad_id, 在batch解码时候使用
        - pad_mode: str, padding在前面还是后面, pre或者post
        - device: str, 默认为'cpu'
        - n: int, random_sample时候表示生成的个数; beam_search时表示束宽
        - top_k: int, 这里的topk是指仅保留topk的值 (仅在top_k上进行概率采样)
        - top_p: float, 这里的topp是token的概率阈值设置(仅在头部top_p上进行概率采样)
        - temperature: float, 温度参数, 默认为1, 越小结果越确定, 越大结果越多样
        - repetition_penalty: float, 重复的惩罚系数, 越大结果越不重复
        - min_ends: int, 最小的end_id的个数
    :param create_model_at_startup: bool, 是否在启动的时候加载模型, 默认为True
    :param system: Optional[str]=None, 模型使用的system信息, 仅部分模型可用, 且openai api格式的不需要设置该参数

    ### 模式
    :param mode: 命令行, web, api服务模式, Literal['raw', 'cli', 'gradio', 'streamlit', 'openai']
    :param template: 使用的模板, 一般在bert4torch_config.json中无需单独设置, 可自行指定

    ### openai api参数
    :param name: str, 模型名称
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    :param offload_when_nocall: str, 是否在一定时长内无调用就卸载模型，可以卸载到内存和disk两种
    :param max_callapi_interval: int, 最长调用间隔
    :param scheduler_interval: int, 定时任务的执行间隔
    :param api_keys: List[str], api keys的list

    ### Examples:
    ```python
    >>> from bert4torch.pipelines import Chat

    >>> checkpoint_path = "E:/data/pretrain_ckpt/glm/chatglm2-6b"
    >>> generation_config  = {'mode':'random_sample',
    ...                     'max_length':2048, 
    ...                     'default_rtype':'logits', 
    ...                     'use_states':True
    ...                     }
    >>> chat = Chat(checkpoint_path, generation_config=generation_config, mode='cli')
    >>> chat.run()
    ```
    """
    def __init__(self, 
                 # 基类使用
                 checkpoint_path:str, 
                 config_path:str=None,
                 precision:Literal['double', 'float', 'half', 'float16', 'bfloat16', None]=None, 
                 quantization_config:dict=None, 
                 generation_config:dict=None, 
                 create_model_at_startup:bool=True,
                 # cli参数
                 system:str=None,
                 # openapi参数
                 name:str='default', 
                 route_api:str='/chat/completions', 
                 route_models:str='/models', 
                 max_callapi_interval:int=24*3600, 
                 scheduler_interval:int=10*60, 
                 offload_when_nocall:Literal['cpu', 'disk']=None, 
                 api_keys:List[str]=None,
                 # 模式
                 mode:Literal['raw','gradio', 'streamlit', 'openai']='raw',
                 template: str=None,
                 **kwargs
                 ) -> None:
        pass

    def __new__(cls, *args, mode:Literal['raw', 'gradio', 'streamlit', 'openai']='raw', **kwargs):
        # template指定使用的模板
        if kwargs.get('template') is not None:
            template = kwargs.pop('template')
        else:
            config_path = kwargs['config_path'] if kwargs.get('config_path') is not None else args[0]
            config = json.load(open(get_config_path(config_path, allow_none=True)))
            template = config.get('template', config.get('model', config.get('model_type')))
        if template is None:
            raise ValueError('template/model/model_type not found in bert4torch_config.json')
        else:
            ChatTemplate = MAPPING[template]
            log_info_once(f'Chat pipeline use template=`{template}` and mode=`{mode}`')

        if mode == 'gradio':
            @add_start_docstrings(CHAT_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVWebGradio): pass
        elif mode == 'streamlit':
            @add_start_docstrings(CHAT_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVWebStreamlit): pass
        elif mode == 'openai':
            @add_start_docstrings(OPENAI_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVOpenaiApi): pass
        elif mode == 'raw':
            ChatDemo = ChatTemplate
        else:
            raise ValueError(f'Unsupported mode={mode}')
        return ChatDemo(*args, **kwargs)