#! -*- coding: utf-8 -*-
# Reward model

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import ListDataset, sequence_padding, DottableDict
from bert4torch.callbacks import Callback, Logger
from bert4torch.optimizers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import get_model_config, get_nbit_lora_model


# 基本参数
args = DottableDict()
args.lr = 1e-5
args.batch_size = 4
args.eval_batch_size = 4
args.grad_accumulation_steps = 1
args.max_seq_length = 512
args.epochs = 1
args.steps_per_epoch = 100
args.use_lora = False
args.load_in_nbit = None
args.data_path = 'E:/Github/MedicalGPT/data/reward/**/*.json'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.model_type, args.dir_path, args.config_path, args.checkpoint_path = get_model_config('bloom')

tokenizer = AutoTokenizer.from_pretrained(args.dir_path, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or -100

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        examples = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    examples.append(json.loads(l))
        new_examples = []
        for example in examples:
            tokenized_chosen = tokenizer.encode("Question: " + example["question"] + "\n\nAnswer: " + example["response_chosen"], max_length=args.max_seq_length)
            tokenized_rejected = tokenizer.encode("Question: " + example["question"] + "\n\nAnswer: " + example["response_rejected"], max_length=args.max_seq_length)
            new_examples.append((tokenized_chosen, tokenized_rejected))
        return new_examples

def collate_fn(batch):
    input_ids_chosen, input_ids_rejected = [], []
    for input_ids_chosen_i, input_ids_rejected_i in batch:
        input_ids_chosen.append(input_ids_chosen_i)
        input_ids_rejected.append(input_ids_rejected_i)
    # padding在左侧
    input_ids_chosen = torch.tensor(sequence_padding(input_ids_chosen, value=pad_token_id, mode='pre'), dtype=torch.long, device=args.device)
    input_ids_rejected = torch.tensor(sequence_padding(input_ids_rejected, value=pad_token_id, mode='pre'), dtype=torch.long, device=args.device)
    return [input_ids_chosen, input_ids_rejected], None

train_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

# 建立模型，加载权重
class Model(BaseModel):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.model = build_transformer_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, model=args.model_type, pad_token_id=pad_token_id, with_lm=False)
        self.score = nn.Linear(self.model.config['hidden_size'], 1)
    
    def forward(self, input_ids_chosen, input_ids_rejected):
        # 最后一个token的logit计算得分，作为整个response的得分
        chosen_score = self.score(self.model(input_ids_chosen)[:, -1, :])
        reject_score = self.score(self.model(input_ids_rejected)[:, -1, :])
        return chosen_score, reject_score

    def predict_score(self, input_ids):
        # 返回某个query+response组合的得分
        return self.score(self.model(input_ids)[:, -1, :])

model = Model()
model = get_nbit_lora_model(model, use_lora=args.use_lora, load_in_nbit=args.load_in_nbit).to(args.device)

# lora
if args.use_lora:
    from peft import LoraConfig
    peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['q', 'k', 'v']
        )
    model = model.get_peft_model(peft_config).to(args.device)
else:
    model = model.to(args.device)

class Loss:
    def __call__(self, output, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''
        rewards_chosen, rewards_rejected = output
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        return loss

optimizer = optim.AdamW(model.parameters(), args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, args.steps_per_epoch*args.epochs)
model.compile(loss=Loss(), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=args.grad_accumulation_steps, clip_grad_norm=1.0)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        perf = self.evaluate(dev_dataloader)
        if self.best < perf['mse']:
            model.save_weights(f'./best_model_reward.pt', trainable_only=True)
            perf['best_mse'] = perf['mse']
        print(perf)
    
    def evaluate(self, data):
        for (input_ids_chosen, input_ids_rejected), _ in tqdm(data, desc='Evaluating'):
            chosen_score, reject_score = model.predict([input_ids_chosen, input_ids_rejected])
            if isinstance(chosen_score, torch.Tensor):
                chosen_score = chosen_score.detach().cpu().numpy()
            if isinstance(reject_score, torch.Tensor):
                reject_score = reject_score.detach().cpu().numpy()
            # MSE
            mse = mean_squared_error(reject_score, chosen_score)
            # MAE
            mae = mean_absolute_error(reject_score, chosen_score)
        return {"mse": mse, "mae": mae}


if __name__ == '__main__':
    evaluator = Evaluator()
    logger = Logger('./log_reward.log')
    model.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[evaluator, logger])
else:
    model.load_weights('./best_model_reward.pt', strict=False)
