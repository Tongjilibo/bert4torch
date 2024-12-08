#! -*- coding: utf-8 -*-
"""
基本测试: 原生llama模型的测试
"""

from bert4torch.models import build_transformer_model
import re


model_dir = 'E:/data/pretrain_ckpt/llama/Llama-3.2-11B-Vision-Instruct'

build_transformer_model(checkpoint_path=model_dir)