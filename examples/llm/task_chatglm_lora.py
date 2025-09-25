#! -*- coding: utf-8 -*-
# chatglm/chatglm2的指令微调, 基于lora/qlora
# peft和transformer包是耦合的，因此这里用法和hf的略有不同
# 参考项目：lora: https://github.com/mymusise/ChatGLM-Tuning
#         qlora: https://github.com/shuxueslpi/chatGLM-6B-QLoRA

# |                   模型                  |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
# | ------------------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
# | chatglm+b4t+lora+V100-fp16-bs16         |  28G      |     2570     |     24.89     |    31.38    |     7.17    |    8.15   |         |
# | chatglm+b4t+qlora+V100-bs16             |  26G      |     5381     |     23.99     |    29.52    |     6.47    |    7.74   |         |

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import ListDataset
from bert4torch.generation import SeqGeneration
from bert4torch.callbacks import Callback, Logger
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.losses import CausalLMLoss
from transformers import AutoTokenizer
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, prepare_model_for_kbit_training  # 需要pip install git+https://github.com/huggingface/peft.git
import os


# ====================================基本参数====================================
model_name = 'chatglm2'  # 可选chatglm, chatglm2
mode = 'train'
load_in_nbit = None  # 量化, 可选None, 8, 4
max_source_length = 64
max_target_length = 64
max_seq_length = max_source_length + max_target_length
lr = 5e-4
batch_size = 16  # 根据显存大小调整
eval_batch_size = 4
grad_accumulation_steps = 1  # 根据显存大小调整
epochs = 1
steps_per_epoch = 3000
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None
data_dir = 'F:/data/corpus/sft/AdvertiseGen'  # 数据路径
if model_name == 'chatglm2':
    model_dir = "E:/data/pretrain_ckpt/THUDM/chatglm2-6b"
elif model_name == 'chatglm':
    model_dir = "E:/data/pretrain_ckpt/THUDM/chatglm-6b"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ====================================加载数据集====================================
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                prompt, response = l[prompt_column], l[response_column]
                history = l.get('history_column', None)
                D.append((prompt, response, history))
        return D

if model_name == 'chatglm':
    def build_prompt(query, history):
        if history_column is None:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, answer) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, answer)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt

    def collate_train_fn(batch):
        batch_token_ids, batch_labels = [], []
        for query, answer, history in batch:
            prompt = build_prompt(query, history)
            prompt = prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[:max_source_length - 1]

            if len(b_ids) > max_target_length - 2:
                b_ids = b_ids[:max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [tokenizer.pad_token_id] * context_length + input_ids[mask_position+1:]
            batch_token_ids.append(input_ids)
            batch_labels.append(labels)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
        batch_labels = torch.tensor(sequence_padding(batch_labels, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
        return [batch_token_ids], batch_labels

elif model_name == 'chatglm2':
    def build_prompt(query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def collate_train_fn(batch):
        batch_token_ids, batch_labels = [], []
        for query, answer, history in batch:
            prompt = build_prompt(query, history)
            prompt = prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, max_length=max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
            batch_token_ids.append(input_ids)
            batch_labels.append(labels)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
        batch_labels = torch.tensor(sequence_padding(batch_labels, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
        return [batch_token_ids], batch_labels

def collate_dev_fn(batch):
    batch_prompt, batch_labels = [], []
    for query, labels, history in batch:
        batch_prompt.append(prefix + build_prompt(query, history))
        
        label_ids = tokenizer(text_target=labels, max_length=max_target_length, truncation=True)['input_ids']
        batch_labels.append(tokenizer.decode(label_ids, skip_special_tokens=True))
    return batch_prompt, batch_labels

train_dataloader = DataLoader(MyDataset(os.path.join(data_dir, 'train.json')), batch_size=batch_size, shuffle=True, collate_fn=collate_train_fn) 
dev_dataloader = DataLoader(MyDataset(os.path.join(data_dir, 'dev.json')), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_dev_fn)


# ====================================建立模型====================================
# 原使用peft=0.5.0时候下面可.half()，高版本peft.half()发现loss为nan，排查发现是高版本会把lora_A转成和base_layer(Linear)的dtype=fp16
# 把.half()去掉，使用原来的bf16训练，lora_A还是fp32
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir, add_trainer=True, tie_word_embeddings=True)

# 量化
if load_in_nbit == 8:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = model.quantize(quant_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
    
elif load_in_nbit == 4: 
    model = model.quantize(
        quant_method='load_in_4bit', 
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
        llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
        )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# lora
peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q', 'k', 'v']
    )
model = model.get_peft_model(peft_config).to(device)

optimizer = optim.AdamW(model.parameters(), lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch*epochs)  # torch4keras<0.0.8需要设置为(steps_per_epoch*epochs)//grad_accumulation_steps
model.compile(loss=CausalLMLoss(offset=True, ignore_index=tokenizer.pad_token_id), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=grad_accumulation_steps, clip_grad_norm=1.0)

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer(text, max_length=max_source_length, truncation=True)['input_ids']]
    def post_process(self, output_ids):
        return [tokenizer.decode(output_id.cpu().numpy()) for output_id in output_ids]
generation = Chat(model, tokenizer, bos_token_id=None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, 
                  mode='random_sample', max_length=512, default_rtype='logits', use_states=True)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        model.save_weights(f'./model.pt', trainable_only=True)
        # # 可以每个epoch都evaluate，但是比较耗时
        # score_dict = self.evaluate(dev_dataloader, epoch)
        # # 保存最优
        # if score_dict['bleu-4'] > self.best:
        #     self.best = score_dict['bleu-4']
        #     model.save_weights('./best_model.pt', trainable_only=True)  # 仅保存lora权重
        # score_dict['best'] = self.best
        # print(score_dict)
    
    def evaluate(self, data, epoch='final'):
        preds, labels = [], []
        for prompt, label in tqdm(data, desc='Evaluating'):
            pred = generation.generate(prompt, top_k=50, top_p=0.7, temperature=0.95)
            preds.extend(pred)
            labels.extend(label)
            with open(f'./preds_{epoch}.txt', 'a+', encoding='utf-8') as f:
                for pred_i, label_i in zip(pred, label):
                    f.write(json.dumps({'pred': pred_i, 'label': label_i}, ensure_ascii=False) + '\n')

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict


if __name__ == '__main__':
    evaluator = Evaluator()
    logger = Logger('./log.log', interval=100)

    if mode == 'train':
        model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[evaluator, logger])
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)

    else:
        model.load_weights('./model.pt', strict=False)
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)
