#! -*- coding: utf-8 -*-
# llama系列的指令微调, 基于lora/qlora，包含多个llama系列模型的指令微调
# peft和transformer包是耦合的，因此这里用法和hf的略有不同
# 参考项目：lora: https://github.com/mymusise/ChatGLM-Tuning
#         qlora: https://github.com/shuxueslpi/chatGLM-6B-QLoRA