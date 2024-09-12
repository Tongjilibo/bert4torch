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
    is_transformers_available
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
    'ChatV',
    ]


class ChatV(ChatBase):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.terminators = ['<|im_end|>', '<|endoftext|>']
    
    def _decode(self, inputs_embeds, attention_mask, **kwargs):
        terminators = [self.tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.model.llm.generate(
            inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            **kwargs
        )
        return output

    def _decode_stream(self, inputs_embeds, attention_mask, **kwargs):
        terminators = [self.tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        generation_kwargs = {
            'text': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'attention_mask': attention_mask,
        }
        generation_kwargs.update(kwargs)

        for output in self.model.llm.stream_generate(inputs_embeds,  pad_token_id=0, eos_token_id=terminators,
                                                     attention_mask=attention_mask, tokenizer=self.tokenizer, **kwargs):
            yield output

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.model.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], attention_mask, **kwargs)
            else:
                result = self._decode(model_inputs["inputs_embeds"], attention_mask, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states
        
        return result

    def chat(
        self,
        image,
        msgs,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image
        
        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if not hasattr(self, 'processor'):
                self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, trust_remote_code=True)
            processor = self.processor
        
        assert self.model.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.model.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs        

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt", 
            max_length=max_inp_length
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
            
        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config
            )
        
        if stream:
            def stream_gen():
                for text in res:
                    text = text if batched else text[0]
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()

        else:
            return res