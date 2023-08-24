# -*- coding: utf-8 -*-
"""
rlhf
正在调试中
"""

from glob import glob
from torch import nn
from bert4torch.snippets import DottableDict
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.generation import SeqGeneration
import torch
from trl import PPOConfig, PPOTrainer, set_seed
from utils import get_model_config
from transformers import AutoTokenizer
from bert4torch.callbacks import Callback, Logger


steps_per_epoch = None
epochs = 1
args = DottableDict()
args.use_fast_tokenizer = False
device = "cuda" if torch.cuda.is_available() else "cpu"


# Set seed before initializing value head for deterministic eval
set_seed(args.seed)

# 模型配置
args.model_type, dir_path, config_path, checkpoint_path = get_model_config('bloom')

def main():
    # =============== 定义tokenizer ==================
    if args.model_type == 'bloom':
        args.use_fast_tokenizer = True
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer  = AutoTokenizer.from_pretrained(dir_path, **tokenizer_kwargs)
    # Required for llama
    if args.model_type == "llama" and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_token_id = tokenizer.pad_token_id or -100

    # =============== 定义Dataset ==================
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for conversation in examples['conversations']:
            for message in conversation:
                instruction = message['value']
                input = message['from']
                if input:
                    instruction = instruction + "\n" + input
                source = PROMPT_TEMPLATE.format_map({"instruction": instruction})
                tokenized_question = tokenizer(
                    source, truncation=True, max_length=max_source_length, padding="max_length",
                    return_tensors="pt"
                )
                new_examples["query"].append(source)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples



    # ============= 定义 model =============
    class Model(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.module = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=args.model_type, 
                                                 pad_token_id=pad_token_id, with_lm=False)
            self.score = nn.Linear(self.module.config['hidden_size'], 1)
        
        def forward(self, input_ids):
            logit = self.score(self.module(input_ids))
            return logit
    model = Model().to(device)

    # ============= 定义reward model =============
    class RewardModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.module = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=args.model_type, 
                                                 pad_token_id=pad_token_id, with_lm=False)
            self.score = nn.Linear(self.module.config['hidden_size'], 1)
        
        def forward(self, input_ids):
            # 最后一个token的logit计算得分，作为整个response的得分
            score = self.score(self.module(input_ids)[:, -1, :])
            return score
    reward_model = RewardModel().to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path, **tokenizer_kwargs)

    output_dir = args.output_dir
    config = PPOConfig(
        steps=args.max_steps,
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        seed=args.seed,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
        project_kwargs={"logging_dir": output_dir},
    )

    # ============= generation =============
    generation_kwargs = {
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "do_sample": True,
    }
    tokenizer_config = {'skip_special_tokens': True, 'add_special_tokens': False}
    generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', tokenizer_config=tokenizer_config,
                            maxlen=max_target_length, default_rtype='logits', use_states=True)


    # ============= PPOTrainer =============
    trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )


    def save_model(save_dir):
        trainer.accelerator.unwrap_model(trainer.model).save_pretrained(save_dir)
        trainer.tokenizer.save_pretrained(save_dir)

    logger = Logger('./log_reward.log')
    trainer.fit(trainer.dataloader, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[logger])

if __name__ == "__main__":
    main()
