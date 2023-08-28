#! -*- coding: utf-8 -*-
# continue pretrain

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import IterDataset, DottableDict
from bert4torch.callbacks import Callback, Logger
from bert4torch.optimizers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob
from utils import get_model_config, get_nbit_lora_model


# 基本参数
args = DottableDict()
args.lr = 5e-5
args.batch_size = 1
args.eval_batch_size = 4
args.grad_accumulation_steps = 4
args.max_seq_length = 512
args.epochs = 1
args.steps_per_epoch = 500
args.data_path = 'E:/Github/MedicalGPT/data/pretrain/**/*.txt'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.use_lora = False
args.load_in_nbit = None
args.model_type, args.dir_path, args.config_path, args.checkpoint_path = get_model_config('bloom')

tokenizer = AutoTokenizer.from_pretrained(args.dir_path, trust_remote_code=True)

# 加载数据集, 数据量较大使用IterDataset
class MyDataset(IterDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    input_ids = tokenizer.encode(text=l, add_special_tokens=False)
                    if len(input_ids) > 0 and input_ids[-1] == tokenizer.eos_token_id:
                        input_ids = input_ids[:-1]
                    if len(D) + len(input_ids) > args.max_seq_length-1:  # +当前输入超长的话，则返回之前的累计输入
                        D += [tokenizer.eos_token_id]
                        yield D
                        D = input_ids
                    else:
                        D.extend(input_ids)

def collate_fn(batch_token_ids):
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=args.device)
    return [batch_token_ids], batch_token_ids

train_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

# 建立模型，加载权重
model = build_transformer_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, model=args.model_type, add_trainer=True)
model = get_nbit_lora_model(model, use_lora=args.use_lora, load_in_nbit=args.load_in_nbit).to(args.device)

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

optimizer = optim.AdamW(model.parameters(), args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, args.steps_per_epoch*args.epochs)
model.compile(loss=CrossEntropyLoss(ignore_index=tokenizer.pad_token_id), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=args.grad_accumulation_steps, clip_grad_norm=1.0)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        acc = self.evaluate(dev_dataloader)
        if self.best < acc['acc']:
            model.save_weights(f'./best_model_pretain.pt', trainable_only=True)
            acc['best_acc'] = acc['acc']
        print(acc)
    
    def evaluate(self, data):
        correct, total = 0, 0
        for input_ids, label in tqdm(data, desc='Evaluating'):
            pred = model.predict(input_ids).argmax(dim=-1)
            label = label[:, 1:]
            pred = pred[:, :-1]
            mask = (label != tokenizer.pad_token_id)
            correct += ((label==pred) * mask).sum().item()
            total += mask.sum().item()

        return {'acc': correct/total}


if __name__ == '__main__':
    evaluator = Evaluator()
    logger = Logger('./log_pretrain.log')
    model.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[evaluator, logger])
else:
    model.load_weights('./best_model_pretain.pt', strict=False)
