#! -*- coding: utf-8 -*-
# llama系列的指令微调, 基于lora/qlora, deepspeed

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, DeepSpeedTrainer
from bert4torch.snippets import ListDataset
from bert4torch.generation import SeqGeneration
from bert4torch.callbacks import Callback, Logger
from bert4torch.optimizers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import pandas as pd
from peft import LoraConfig, prepare_model_for_kbit_training
import os


# 基本参数
mode = 'train'
max_source_length = 256
max_target_length = 256
eval_batch_size = 4
max_seq_length = max_source_length + max_target_length
epochs = 1
prefix = ''

# 模型配置
dir_path = '/mnt/e/pretrain_ckpt/llama-2/llama-2-7b-chat'
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
tokenizer.pad_token_id = 0

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        df = pd.read_csv(filename)
        df = df['text'].str.replace('<s>Human: ', '').str.replace('\n</s>', '').str.split('<s>Assistant: ')

        D = []
        for i in range(len(df)):
            prompt, response = df[i][0], df[i][1]
            D.append((prompt, response, []))
        return D

def build_prompt(query, answer=None, history=[]):
    prompt = ""
    for old_query, old_answer in history:
        prompt += "<s>Human: {}\n</s><s>Assistant: {}\n</s>".format(old_query, old_answer)
    prompt += "<s>Human: {}\n</s><s>Assistant: ".format(query)
    
    if answer is not None:
        prompt += answer + "\n</s>"
    return prompt

def collate_train_fn(batch):
    batch_token_ids = []
    for query, answer, history in batch:
        prompt = prefix + build_prompt(query, answer, history)
        token_ids = tokenizer(text_target=prompt, max_length=max_source_length, truncation=True)['input_ids']
        batch_token_ids.append(token_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
    return [batch_token_ids], batch_token_ids

def collate_dev_fn(batch):
    batch_prompt, batch_labels = [], []
    for query, labels, history in batch:
        batch_prompt.append(prefix + build_prompt(query, None, history))
        
        label_ids = tokenizer(text_target=labels, max_length=max_target_length, truncation=True)['input_ids']
        batch_labels.append(tokenizer.decode(label_ids, skip_special_tokens=True))
    return batch_prompt, batch_labels


# 建立模型，加载权重
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama', add_trainer=True).half()

# 量化
load_in_nbit = None
if load_in_nbit == 8:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model = model.quantize(quantization_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
    model.lm_head = CastOutputToFloat(model.lm_head)
    
elif load_in_nbit == 4:
    from transformers import BitsAndBytesConfig
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
                                llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
                                )
    model = model.quantize(quantization_method='load_in_4bit', quantization_config=q_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# lora
peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q', 'k', 'v']
    )
model = model.get_peft_model(peft_config)
model_ds = DeepSpeedTrainer(model, config_path='./deepspeed.json')

batch_size = model_ds.config.train_micro_batch_size_per_gpu
train_dataloader = DataLoader(MyDataset('/mnt/e/data/corpus/prompt/Llama2-Chinese/train_sft.csv'), batch_size=batch_size, shuffle=True, collate_fn=collate_train_fn) 
dev_dataloader = DataLoader(MyDataset('/mnt/e/data/corpus/prompt/Llama2-Chinese/dev_sft.csv'), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_dev_fn)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, logits, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''
        raw_dtyps = logits.dtype
        logits = logits.to(torch.float32)
        logits = logits[:, :-1, :].contiguous()  # 预测序列，错开一位
        labels = labels[:, 1:].contiguous() # 目标token_ids
        
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.flatten()
        loss = super().forward(logits, labels)

        return loss.to(raw_dtyps)

model_ds.compile(loss=CrossEntropyLoss(ignore_index=tokenizer.pad_token_id), optimizer=None)

tokenizer_config = {'skip_special_tokens': True, 'add_special_tokens': False}
generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', tokenizer_config=tokenizer_config,
                           maxlen=max_seq_length, default_rtype='logits', use_states=True)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        model_ds.save_weights(f'./model.pt', trainable_only=True)
    
    def evaluate(self, data, epoch='final'):
        preds, labels = [], []
        for prompt, label in tqdm(data, desc='Evaluating'):
            pred = generation.batch_generate(prompt, topk=50, topp=0.7, temperature=0.95)
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
    logger = Logger('./log.log')
    logger.run_callback = model_ds.deepspeed_engine.local_rank == 0  # 指定只有local_rank=0的记录log文件

    if mode == 'train':
        model_ds.fit(train_dataloader, steps_per_epoch=None, epochs=epochs, callbacks=[evaluator, logger])
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)

    else:
        model_ds.load_weights('./model.pt', strict=False)
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)
