#! -*- coding: utf-8 -*-
# Supervised Finetune

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, DottableDict
from bert4torch.callbacks import Callback, Logger
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
import json
from glob import glob
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import get_model_config, get_conv_template, get_nbit_lora_model


# 基本参数
args = DottableDict()
args.max_source_length = 256
args.max_target_length = 256
args.max_length = args.max_source_length + args.max_target_length
args.batch_size = 2
args.grad_accumulation_steps = 4
args.lr = 5e-5
args.epochs = 1
args.use_lora = False
args.load_in_nbit = None
args.data_path = 'E:/Github/MedicalGPT/data/finetune/**/*.jsonl'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.model_name = 'bloom'
args.model_type, args.dir_path, args.config_path, args.checkpoint_path = get_model_config(args.model_name)

tokenizer = AutoTokenizer.from_pretrained(args.dir_path, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or -100

def preprocess_function(examples):
    """
    Preprocessing the datasets.
        part of code modified from https://github.com/lm-sys/FastChat
    """
    input_ids_list = []
    targets_list = []
    roles = ["human", "gpt"]
    prompt_template = get_conv_template(args.model_name)

    def get_dialog(examples):
        for i, source in enumerate(examples):
            if len(source) < 2:
                continue
            data_role = source[0].get("from", "")
            if data_role not in roles or data_role != roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            if len(source) < 2:
                continue
            messages = []
            for j, sentence in enumerate(source):
                data_role = sentence.get("from", "")
                if data_role not in roles:
                    logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                    break
                if data_role == roles[j % 2]:
                    messages.append(sentence["value"])
            if len(messages) < 2 or len(messages) % 2 != 0:
                continue
            # Convert the list to pairs of elements
            history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
            dialog = prompt_template.get_dialog(history_messages)
            yield dialog

    for dialog in get_dialog(examples):
        input_ids, labels = [], []

        for i in range(len(dialog) // 2):
            source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
            target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

            if len(source_ids) > args.max_source_length:
                source_ids = source_ids[:args.max_source_length]
            if len(target_ids) > args.max_target_length - 1:  # eos token
                target_ids = target_ids[:args.max_target_length - 1]
            if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                source_ids = source_ids[1:]
            if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                target_ids = target_ids[:-1]
            if len(input_ids) + len(source_ids) + len(target_ids) + 1 > args.max_length:
                break

            input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
            labels += [pad_token_id] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        input_ids_list.append(input_ids)
        targets_list.append(labels)

    return list(zip(input_ids_list, targets_list))

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    D.append(json.loads(l)['conversations'])
        return preprocess_function(D)

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for token_ids, label_ids in batch:
        batch_token_ids.append(token_ids)
        batch_labels.append(label_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=pad_token_id), dtype=torch.long, device=args.device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=pad_token_id), dtype=torch.long, device=args.device)
    return [batch_token_ids], batch_labels

train_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) 
model = build_transformer_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, model=args.model_type, add_trainer=True, pad_token_id=pad_token_id,
                                grad_accumulation_steps=args.grad_accumulation_steps).to(args.device)
model = get_nbit_lora_model(model, use_lora=args.use_lora, load_in_nbit=args.load_in_nbit).to(args.device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        '''
        y_pred: [btz, seq_len, vocab_size]
        y_true: token_ids: [btz, seq_len]
        '''
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)

loss_fun = CrossEntropyLoss(ignore_index=pad_token_id)
model.compile(loss=loss_fun, optimizer=optim.Adam(model.parameters(), args.lr))

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        dev_loss = self.evaluate(dev_dataloader)
        if dev_loss['dev_loss'] <= self.lowest:
            self.lowest = dev_loss['dev_loss']
            model.save_weights('./best_model_sft.pt')
        dev_loss['best_dev_loss'] = self.lowest
        print(dev_loss)

    def evaluate(self, data):
        loss, count = 0, 0
        for input_ids, label in tqdm(data, desc='Evaluating'):
            pred = model.predict(input_ids)
            loss += loss_fun(pred, label).item()
            count += 1

        return {'dev_loss': loss/count}

if __name__ == '__main__':
    logger = Logger('./log_sft.log')
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=args.epochs, callbacks=[evaluator, logger])
else:
    model.load_weights('./best_model_sft.pt')
