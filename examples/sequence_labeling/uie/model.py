import numpy as np
import torch
import torch.nn as nn
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BERT


config_path = 'E:/data/pretrain_ckpt/Tongjilibo/uie-base/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/Tongjilibo/uie-base/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/Tongjilibo/uie-base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)


class UIE(BERT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output+task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def forward(self, token_ids, token_type_ids):
        outputs = super().forward([token_ids, token_type_ids])
        sequence_output = outputs[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        return start_prob, end_prob
    
    @torch.no_grad()
    def predict(self, token_ids, token_type_ids):
        self.eval()
        start_prob, end_prob = self.forward(token_ids, token_type_ids)
        return start_prob, end_prob

custom_model = False
if custom_model:
    # 使用外部自定义的模型
    uie_model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=UIE, with_pool=True, add_trainer=True)
    print('Load custom uie model done')
else:
    # 使用bert4torch自带的uie
    uie_model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='uie', with_pool=True, add_trainer=True)
    print('Load inner uie model done')