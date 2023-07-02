#! -*- coding: utf-8 -*-
# 权重转换脚本：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_chatglm.py
# chatglm的指令微调, 基于ptuning_v2，性能和官方项目给出的指标相当
# |            chatglm              |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
# | ----------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
# | hf+pt2 official+v100            |   ——      |      ——      |     24.97     |    31.12    |     7.11    |    8.10   |         |
# | hf+pt2 reappear+v100            |   ——      |      ——      |     24.80     |    30.97    |     6.98    |    7.85   |         |
# | b4t+chatglm+ptuning_v2+v100     |   ——      |      ——      |     24.58     |    30.76    |     7.12    |    8.12   |         |
# | b4t+pt2+T4-int8-bs1             |  10G      |     1470     |     24.87     |    30.83    |     7.14    |    8.05   |         |
# | b4t+pt2+A100(pcie 40G)-fp16-bs1 |  15G      |     287      |     25.10     |    31.43    |     7.30    |    8.28   |         |
# | b4t+pt2+A100(pcie 40G)-fp16-bs8 |  22G      |     705      |     25.22     |    31.22    |     7.38    |    8.35   |         |
# | b4t+pt2+A100(pcie 40G)-fp32-bs1 |  29G      |     760      |     24.83     |    30.95    |     7.18    |    8.08   |         |
# | b4t+pt2+A100(pcie 40G)-fp32-bs4 |  32G      |     2600     |     25.12     |    31.55    |     7.21    |    8.02   |         |

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.callbacks import Callback
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from transformers import AutoTokenizer
from bert4torch.snippets import ListDataset
from bert4torch.callbacks import Logger
from bert4torch.generation import SeqGeneration
from bert4torch.optimizers import get_linear_schedule_with_warmup
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm

# 基本参数
mode = 'train'
max_source_length = 64
max_target_length = 64
lr = 2e-2
batch_size = 1
eval_batch_size = 4
grad_accumulation_steps = 16
steps_per_epoch = 3000
epochs = 16  # torch4keras>=0.0.8后需要设置为1，因为改为了1个batch_step包含了grad_accumulation_steps
max_seq_length = max_source_length + max_target_length
ignore_pad_token_for_loss = True
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None
use_states = True

# 模型配置
choice = 'int4'  # default, int4, int8
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/chatglm/6B"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/chatglm/6B-int4"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/chatglm/6B-int8"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'

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
        batch_prompt.append(prefix + build_prompt(query, history))
        
        label_ids = tokenizer(text_target=labels, max_length=max_target_length, truncation=True)['input_ids']
        batch_labels.append(tokenizer.decode(label_ids, skip_special_tokens=True))
    return batch_prompt, batch_labels

train_dataloader = DataLoader(MyDataset('E:/data/corpus/prompt/AdvertiseGen/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_train_fn) 
dev_dataloader = DataLoader(MyDataset('E:/data/corpus/prompt/AdvertiseGen/dev.json'), batch_size=eval_batch_size, shuffle=False, collate_fn=collate_dev_fn)

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_hidden_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 建立模型，加载权重
        if choice == 'default':
            self.encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half().quantize(4, target_modules=['q', 'k', 'v', 'o', 'intermediateDense', 'outputDense']).to(device)
        else:
            # 在config中已经写入了量化的配置参数
            self.encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').to(device)
        self.config = self.encoder.configs
        self.config.pre_seq_len = 128
        self.config.prefix_projection = False
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

    def get_past_key_values(self, token_ids):
        batch_size = token_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(token_ids.device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(torch.float16)
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        # b, nh, seq_len, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values
    
    def forward(self, token_ids):
        past_key_values = self.get_past_key_values(token_ids)
        logits = self.encoder([token_ids], past_key_values=past_key_values)
        return logits
    
    @torch.no_grad()
    def predict(self, inputs, **inputs_kwargs):
        self.eval()
        token_ids = inputs[0]
        # use_states=False时候，每次都重新生成past_key_values
        # use_states=True时候，仅在第0步生成past_key_values
        if inputs_kwargs.get('past_key_values', None) is None:
            past_key_values = self.get_past_key_values(token_ids)
            inputs_kwargs['past_key_values'] = past_key_values
        return self.encoder([token_ids],  **inputs_kwargs)
    
model = Model().to(device)
model.print_trainable_parameters()

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

optimizer = optim.AdamW(model.parameters(), lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, (steps_per_epoch*epochs)//grad_accumulation_steps)  # torch4keras>=0.0.8后需要设置为steps_per_epoch*epochs就可以
model.compile(loss=CrossEntropyLoss(ignore_index=-100), optimizer=optimizer, scheduler=scheduler, grad_accumulation_steps=grad_accumulation_steps, clip_grad_norm=1.0)

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer(text, max_length=max_source_length, truncation=True)['input_ids']]
    def post_process(self, output_ids):
        return [tokenizer.decode(output_id.cpu().numpy()) for output_id in output_ids]
generation = Chat(model, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], pad_id=tokenizer.pad_token_id, 
                  mode='random_sample', maxlen=512, default_rtype='logits', use_states=use_states)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        model.save_weights(f'./model_{epoch}.pt', trainable_only=True)
        # # 可以每个epoch都evaluate，但是比较耗时
        # score_dict = self.evaluate(dev_dataloader, epoch)
        # # 保存最优
        # if score_dict['bleu-4'] > self.best:
        #     self.best = score_dict['bleu-4']
        #     model.save_weights('./best_model.pt', trainable_only=True)  # 仅保存PrefixEncoder权重
        # score_dict['best'] = self.best
        # print(score_dict)
    
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
    logger = Logger('./log.log', interval=100)

    if mode == 'train':
        model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[evaluator, logger])
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)

    else:
        model.load_weights('./model_15.pt', strict=False)
        score_dict = evaluator.evaluate(dev_dataloader)
        print(score_dict)
