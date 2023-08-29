# -*- coding: utf-8 -*-
"""
dpo: 仍在测试中
"""

from glob import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
from bert4torch.snippets import DottableDict, ListDataset, sequence_padding
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.generation import SeqGeneration
from bert4torch.callbacks import Callback, Logger
from bert4torch.trainer import PPOTrainerTrl
from trl import PPOConfig, set_seed
from utils import get_model_config, get_nbit_lora_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


# 基本参数
args = DottableDict()
args.steps_per_epoch = None
args.epochs = 1
args.data_path = 'E:/Github/MedicalGPT/data/reward/**/*.json'
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.use_fast_tokenizer = False
args.seed = 1234
args.max_seq_length = 512
args.reward_model_name_or_path = "E:/pretrain_ckpt/deberta/[OpenAssistant]--reward-model-deberta-v3-large-v2"
args.load_in_8bit = False
args.max_steps = 100
args.learning_rate = 1e-5
args.batch_size = 8
args.gradient_accumulation_steps = 1
args.target_kl = 0.1
args.init_kl_coef = 0.2
args.adap_kl_ctrl = True
args.trust_remote_code = True
args.use_lora = False
args.load_in_nbit = None
args.model_type, args.model_name_or_path, args.config_path, args.checkpoint_path = get_model_config('bloom')

set_seed(args.seed)

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


# ============= 定义 model =============
class ActorModel(BaseModel):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.module = build_transformer_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, model=args.model_type, 
                                                pad_token_id=pad_token_id)
        self.score = nn.Linear(self.module.config['hidden_size'], 1)
    
    def forward(self, *args, **kwargs):
        self.module.with_lm = False
        hidden_states = self.module(kwargs['input_ids'])
        lm_logits = self.module.lm_head(hidden_states)
        value = self.score(hidden_states).squeeze(-1)
        return lm_logits, None, value
model = ActorModel().to(args.device)
model = get_nbit_lora_model(model, use_lora=args.use_lora, load_in_nbit=args.load_in_nbit).to(args.device)


# ============= 定义reward model =============

# ============= generation =============
generation_kwargs = {
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "topp": 1.0,
}
tokenizer_config = {'skip_special_tokens': True, 'add_special_tokens': False}
generation = SeqGeneration(model.module, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', tokenizer_config=tokenizer_config,
                           maxlen=max_target_length, default_rtype='logits', use_states=True)


# ============= PPOTrainer =============
config = PPOConfig(
    steps=args.max_steps,
    model_name=args.model_name_or_path,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    target_kl=args.target_kl,
    seed=args.seed,
    init_kl_coef=args.init_kl_coef,
    adap_kl_ctrl=args.adap_kl_ctrl
)

trainer = PPOTrainerTrl(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    reward_model=reward_model,
    reward_tokenizer=reward_tokenizer,
    dataset=train_dataset,
    data_collator=collate_fn,
    generation=generation,
    generation_kwargs=generation_kwargs
)


if __name__ == "__main__":
    logger = Logger('./log_rlhf.log')
    trainer.fit(trainer.dataloader, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[logger])
