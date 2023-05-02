#! -*- coding: utf-8 -*-
# chatglm的指令微调, 基于lora, 还在测试中
# peft和transformer包是耦合的，因此这里用法和hf的略有不同

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
from bert4torch.callbacks import Callback
from transformers import AutoTokenizer
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
from peft import LoraConfig


# 基本参数
max_source_length = 64
max_target_length = 64
lr = 2e-2
batch_size = 1
eval_batch_size = 4
grad_accumulation_steps = 16
max_seq_length = max_source_length + max_target_length
ignore_pad_token_for_loss = True
epochs = 4
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None

# 模型配置
dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)

# 加载数据集
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
        labels = [-100] * context_length + input_ids[mask_position+1:]
        batch_token_ids.append(input_ids)
        batch_labels.append(labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=-100), dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels

def collate_dev_fn(batch):
    batch_prompt, batch_labels = [], []
    for query, labels, history in batch:
        batch_prompt.append(build_prompt(query, history))
        batch_labels.append(labels)
    return batch_prompt, batch_labels

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_train_fn) 
dev_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/dev.json'), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_dev_fn)

peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q', 'k', 'v']
    )

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm', num_hidden_layers=1,
                                token_pad_ids=tokenizer.pad_token_id, add_trainer=True).get_peft_model(peft_config).to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, logits, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''

        logits = logits[:, :-1, :]  # 预测序列，错开一位
        labels = labels[:, 1:]# 目标token_ids
        
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.flatten()
        return super().forward(logits, labels)
model.compile(loss=CrossEntropyLoss(ignore_index=-100), optimizer=optim.Adam(model.parameters(), lr), grad_accumulation_steps=grad_accumulation_steps)

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer.encode(text)]
    def post_process(self, output_ids):
        return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(model, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], mode='random_sample',
                  maxlen=512, default_rtype='logits', use_states=True)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        score_dict = self.evaluate(dev_dataloader)
        # 保存最优
        if score_dict['bleu-4'] > self.best:
            self.best = score_dict['bleu-4']
            model.save_weights('./best_model.pt', trainable_only=True)  # 仅保存lora权重
        score_dict['best'] = self.best
        print(score_dict)
    
    def evaluate(self, data):
        preds, labels = [], []
        for prompt, label in tqdm(data, desc='Evaluating'):
            pred = generation.batch_generate(prompt, topk=50, topp=0.7, temperature=0.95)
            preds.extend(pred)
            labels.extend(label)

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

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt', strict=False)
