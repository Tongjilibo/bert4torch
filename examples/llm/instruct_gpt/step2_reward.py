#! -*- coding: utf-8 -*-
# Reward model

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import ListDataset
from bert4torch.callbacks import Callback, Logger
from bert4torch.optimizers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import get_model_config


# 基本参数
lr = 1e-5
batch_size = 4
eval_batch_size = 4
grad_accumulation_steps = 1
max_seq_length = 512
epochs = 1
steps_per_epoch = 100
use_lora = False
load_in_nbit = None
data_path = 'E:/Github/MedicalGPT/data/reward/**/*.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型配置
model_type, dir_path, config_path, checkpoint_path = get_model_config('bloom')

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
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
            tokenized_chosen = tokenizer.encode("Question: " + example["question"] + "\n\nAnswer: " + example["response_chosen"], max_length=max_seq_length)
            tokenized_rejected = tokenizer.encode("Question: " + example["question"] + "\n\nAnswer: " + example["response_rejected"], max_length=max_seq_length)
            new_examples.append((tokenized_chosen, tokenized_rejected))
        return new_examples

def collate_fn(batch):
    input_ids_chosen, input_ids_rejected = [], []
    for input_ids_chosen_i, input_ids_rejected_i in batch:
        input_ids_chosen.append(input_ids_chosen_i)
        input_ids_rejected.append(input_ids_rejected_i)
    # padding在左侧
    input_ids_chosen = torch.tensor(sequence_padding(input_ids_chosen, value=pad_token_id, mode='pre'), dtype=torch.long, device=device)
    input_ids_rejected = torch.tensor(sequence_padding(input_ids_rejected, value=pad_token_id, mode='pre'), dtype=torch.long, device=device)
    return [input_ids_chosen, input_ids_rejected], None

train_dataloader = DataLoader(MyDataset(glob(data_path, recursive=True)), batch_size=batch_size, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(data_path, recursive=True)), batch_size=eval_batch_size, collate_fn=collate_fn)

# 建立模型，加载权重
class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_type, pad_token_id=pad_token_id, with_lm=False)
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

# 量化
if load_in_nbit == 8:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = model.quantize(quantization_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
    
elif load_in_nbit == 4:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
                                llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
                                )
    model = model.quantize(quantization_method='load_in_4bit', quantization_config=q_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# lora
if use_lora:
    from peft import LoraConfig
    peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['q', 'k', 'v']
        )
    model = model.get_peft_model(peft_config).to(device)
else:
    model = model.to(device)

class Loss:
    def __call__(self, output, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''
        rewards_chosen, rewards_rejected = output
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        return loss

optimizer = optim.AdamW(model.parameters(), lr)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch*epochs)
model.compile(loss=Loss(), optimizer=optimizer, scheduler=scheduler, 
              grad_accumulation_steps=grad_accumulation_steps, clip_grad_norm=1.0)

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
    model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[evaluator, logger])
else:
    model.load_weights('./best_model_reward.pt', strict=False)
