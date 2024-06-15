#! -*- coding: utf-8 -*-
# chatglm/chatglm2的指令微调, 基于ptuning_v2，性能和官方项目给出的指标相当
# |                 model                   |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
# | ------------------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
# | chatglm+hf+pt2 official+v100-int4-bs1   |   ——      |      ——      |     24.97     |    31.12    |     7.11    |    8.10   |         |
# | chatglm+hf+pt2 reappear+v100-int4-bs1   |   ——      |      ——      |     24.80     |    30.97    |     6.98    |    7.85   |         |
# | chatglm+b4t+pt2+v100+int4+bs1           |   ——      |      ——      |     24.58     |    30.76    |     7.12    |    8.12   |         |
# | chatglm+b4t+pt2+T4-int8-bs1             |  10G      |     1470     |     24.87     |    30.83    |     7.14    |    8.05   |         |
# | chatglm+b4t+pt2+A100(pcie 40G)-fp16-bs1 |  15G      |     287      |     25.10     |    31.43    |     7.30    |    8.28   |         |
# | chatglm+b4t+pt2+A100(pcie 40G)-fp16-bs8 |  22G      |     705      |     25.22     |    31.22    |     7.38    |    8.35   |         |
# | chatglm+b4t+pt2+A100(pcie 40G)-fp32-bs1 |  29G      |     760      |     24.83     |    30.95    |     7.18    |    8.08   |         |
# | chatglm+b4t+pt2+A100(pcie 40G)-fp32-bs4 |  32G      |     2600     |     25.12     |    31.55    |     7.21    |    8.02   |         |
# | chatglm2+b4t+pt2+v100+int4+bs1          |   7G      |      ——      |     24.36     |    29.97    |     6.66    |    7.89   |         |

from bert4torch.snippets import sequence_padding
from bert4torch.callbacks import Callback
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from transformers import AutoTokenizer
from bert4torch.snippets import ListDataset, seed_everything
from bert4torch.callbacks import Logger
from bert4torch.generation import SeqGeneration
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.trainer import PtuningV2Trainer
from bert4torch.losses import CausalLMLoss
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import os


# ====================================基本参数====================================
mode = 'train'  # train evaluate inference
model_name = 'chatglm2-6B'  # chatglm-6B, chatglm-6B-int4, chatglm-6B-int8, chatglm2-6B, chatglm2-6B-int4, chatglm2-6B-int8
max_source_length = 64
max_target_length = 64
max_seq_length = max_source_length + max_target_length
lr = 2e-2
batch_size = 1
eval_batch_size = 16
grad_accumulation_steps = 16
steps_per_epoch = 3000
epochs = 1  # torch4keras<0.0.8后需要设置为16，因为1个batch_step不包含grad_accumulation_steps
ignore_pad_token_for_loss = True
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None
use_states = True
data_dir = '/data/corpus/sft/AdvertiseGen'  # 数据路径
model_dir = f"/data/pretrain_ckpt/glm/{model_name}"  # 模型路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything(42)

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

if model_name in {'chatglm-6B', 'chatglm-6B-int8', 'chatglm-6B-int4'}:
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
else:
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
if model_name in {'chatglm-6B', 'chatglm2-6B'}:
    encoder = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).half()
    encoder = encoder.quantize(quantization_method='cpm_kernels', quantization_bit=4, 
                               target_modules=['q', 'k', 'v', 'o', 'intermediateDense', 'outputDense']).to(device)
else:
    # 在config中已经写入了量化的配置参数
    encoder = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to(device)

model = PtuningV2Trainer(encoder)
model.to(device)
model.print_trainable_parameters()

optimizer = optim.AdamW(model.parameters(), lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch*epochs)  # torch4keras<0.0.8需要设置为(steps_per_epoch*epochs)//grad_accumulation_steps
model.compile(loss=CausalLMLoss(offset=True, ignore_index=tokenizer.pad_token_id), optimizer=optimizer, 
              scheduler=scheduler, grad_accumulation_steps=grad_accumulation_steps, clip_grad_norm=1.0)

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer(text, max_length=max_source_length, truncation=True)['input_ids']]
    def post_process(self, output_ids):
        return [tokenizer.decode(output_id.cpu().numpy()) for output_id in output_ids]
generation = Chat(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, 
                  mode='random_sample', maxlen=512, default_rtype='logits', use_states=use_states)

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
        #     model.save_weights('./model.pt', trainable_only=True)  # 仅保存PrefixEncoder权重
        # score_dict['best'] = self.best
        # print(score_dict)
    
    def evaluate(self, data, epoch='final'):
        preds, labels = [], []
        for prompt, label in tqdm(data, desc='Evaluating'):
            pred = generation.generate(prompt, topk=50, topp=0.7, temperature=0.95)
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

    elif mode == 'evaluate':
        model.load_weights('./model.pt', strict=False)
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)

    else:
        model.load_weights('./model.pt', strict=False)
        query = '类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤'
        for response in generation.stream_generate(query, topk=50, topp=0.7, temperature=0.95):
            os.system('clear')
            print(response[0], flush=True)