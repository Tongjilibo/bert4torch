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
from bert4torch.pipelines.chat import ChatBase
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
    inference_mode
)
from packaging import version
import gc
import time
import json
import requests
from contextlib import asynccontextmanager
import threading
import re
import copy
from argparse import REMAINDER, ArgumentParser
from copy import deepcopy
from PIL import Image
import inspect


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
    from transformers import AutoProcessor, TextIteratorStreamer

__all__ = [
    'ChatVBase',
    'MiniCPMV'
    ]


class ChatVBase(ChatBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def generate(self, *args, **kwargs):
        '''base模型使用'''
        return self.model.generate(*args, **kwargs)

    def build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @inference_mode()
    def chat(
        self,
        query,
        images,
        history:List[dict]=None,
        vision_hidden_states=None,
        max_inp_length=8192,
        system_prompt='',
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        inputs = self.build_prompt(query, images, history, max_inp_length=max_inp_length, max_slice_nums=max_slice_nums,
                                   system_prompt=system_prompt, use_image_id=use_image_id)
        answer = self.generate(
            **inputs,
            vision_hidden_states=vision_hidden_states,
            **self.generation_config,
            **kwargs
        )
        return answer


class MiniCPMV(ChatVBase):
    def build_prompt(
            self,
            queries: Union[str, List[str]], 
            images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]], 
            history: List[Dict], 
            **kwargs
        ) -> str:
        '''
        history: [
                    {'role': 'user', 'content': '图片中描述的是什么', 'images': [PIL.Image.Image]},
                    {'role': 'assistant', 'content': '该图片中描述了一个小男孩在踢足球'},
                 ]

        |    query    |        images      |     comment      |
        |   -------   |      --------      |    ---------     |
        |     str     |        Image       |    提问单张图片   |
        |     str     |     List[Image]    |  同时提问多张图片  |
        |  List[str]  |        Image       |  多次提问单张图片  |
        |  List[str]  |     List[Image]    |  各自提问单张图片  |
        |  List[str]  |  List[List[Image]] |各自同时提问多张图片|
        '''

        if isinstance(queries, str):
            queries = [queries]
            if isinstance(images, Image.Image):
                # 提问单张图片
                images = [images]
            elif isinstance(images, List) and isinstance(images[0], Image.Image):
                # 同时提问多张图片
                images = [images]
            elif images is None:
                images = [images]
        elif isinstance(queries, List) and isinstance(queries[0], str):
            if isinstance(images, Image.Image):
                # 多次提问单张图片
                images = [images] * len(queries)
            elif isinstance(images, List) and isinstance(images[0], Image.Image):
                # 各自提问单张图片
                pass
            elif isinstance(images, List) and isinstance(images[0], List) and isinstance(images[0][0], Image.Image):
                # 各自同时提问多张图片
                pass

        assert len(queries) == len(images), "The batch dim of query and images should be the same."        
        assert self.model.config.query_num == self.processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.patch_size == self.processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.use_image_id == self.processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.slice_config.max_slice_nums == self.processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        # assert self.model.config.slice_mode == self.processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        # 处理history
        history_images = []
        for i, hist in enumerate(history or []):
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
        for query, image in zip(queries, images):
            copy_msgs = copy.deepcopy(history) or []
            if image is None:
                image = []
            elif isinstance(image, Image.Image):
                image = [image]
            content = "(<image>./</image>)\n"*len(image) + query
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
        return inputs