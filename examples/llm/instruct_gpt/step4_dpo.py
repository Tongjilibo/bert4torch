# -*- coding: utf-8 -*-
"""
dpo算法，可用于替换reward+rlhf两个步骤
"""

from glob import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.snippets import DottableDict, ListDataset, sequence_padding, seed_everything
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.trainer import DPOTrainer
from bert4torch.callbacks import Callback, Logger
from bert4torch.losses import DPOLoss
from utils import get_model_config, get_nbit_lora_model
from transformers import AutoTokenizer
import json
import copy


# 基本参数
args = DottableDict()
args.steps_per_epoch = None
args.epochs = 1
args.data_path = 'E:/Github/MedicalGPT/data/reward/**/*.json'
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.use_fast_tokenizer = False
args.seed = 1234
args.lr = 1e-5
args.batch_size = 2
args.max_src_length = 128
args.max_tgt_length = 128
args.full_max_length = args.max_src_length + args.max_tgt_length
args.grad_accumulation_steps = 1
args.trust_remote_code = True
args.use_lora = False
args.load_in_nbit = None
args.model_type, args.model_name_or_path, args.config_path, args.checkpoint_path = get_model_config('bloom')

seed_everything(args.seed)

# =============== 定义tokenizer ==================
if args.model_type == 'bloom':
    args.use_fast_tokenizer = True
# Load tokenizer
tokenizer_kwargs = {
    "use_fast": args.use_fast_tokenizer,
    "trust_remote_code": args.trust_remote_code,
}
tokenizer  = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
# Required for llama
if args.model_type == "llama" and tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_token_id = tokenizer.pad_token_id or -100


# =============== 定义Dataset ==================
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
            prompt_ids = tokenizer.encode("Question: " + example["question"] + "\n\nAnswer: ", max_length=args.max_src_length, add_special_tokens=False)
            chosen_ids = tokenizer.encode(example["response_chosen"], max_length=args.max_tgt_length, add_special_tokens=False) + [tokenizer.eos_token_id]
            rejected_ids = tokenizer.encode(example["response_rejected"], max_length=args.max_tgt_length, add_special_tokens=False) + [tokenizer.eos_token_id]
            new_examples.append((prompt_ids, chosen_ids, rejected_ids))
        return new_examples

def collate_fn(batch):
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
    for prompt_id, chosen_id, rejected_id in batch:
        chosen_ids.append(prompt_id+chosen_id)
        chosen_labels.append([pad_token_id]*len(prompt_id) + chosen_id)  # prompt部分用padding位
        rejected_ids.append(prompt_id+rejected_id)
        rejected_labels.append([pad_token_id]*len(prompt_id) + rejected_id)
    # 这里是把chosen和rejected放到同一个batch中，前半部分是chosen，后半部分是rejected
    input_ids = torch.tensor(sequence_padding(chosen_ids+rejected_ids, value=pad_token_id), dtype=torch.long, device=args.device)
    input_labels = torch.tensor(sequence_padding(chosen_labels+rejected_labels, value=pad_token_id), dtype=torch.long, device=args.device)
    return input_ids, input_labels

train_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(args.data_path, recursive=True)), batch_size=args.batch_size, collate_fn=collate_fn)


# ============= 定义 model =============
net = build_transformer_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, 
                                model=args.model_type, pad_token_id=pad_token_id).to(args.device)
trainer = DPOTrainer(net)
trainer.compile(loss=DPOLoss(pad_token_id=pad_token_id, offset=True), 
                optimizer=optim.AdamW(net.parameters(), args.lr), 
                grad_accumulation_steps=args.grad_accumulation_steps)


if __name__ == "__main__":
    logger = Logger('./log_dpo.log')
    trainer.fit(train_dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[logger])
