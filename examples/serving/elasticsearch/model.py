#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, get_pool_emb
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

# 加载模型，请更换成自己的路径
root_model_path = "F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)

class BertClient(object):
    def __init__(self, batch_size=10, maxlen=128, use_tqdm=False):
        self.model = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0, with_pool=True, output_all_encoded_layers=True)  # 建立模型，加载权重
        self.model.to(device)
        self.model.eval()
        self.batch_size=batch_size
        self.maxlen = maxlen
        self.use_tqdm = use_tqdm

    @torch.no_grad()
    def encode(self, queries):
        token_ids = tokenizer.encode(queries, maxlen=self.maxlen)[0]
        token_ids = torch.tensor(sequence_padding(token_ids), device=device)
        dataloader = DataLoader(TensorDataset(token_ids), batch_size=self.batch_size)

        reps = []
        dataloader = tqdm(dataloader) if self.use_tqdm else dataloader
        for batch in dataloader:
            hidden_state1, pooler = self.model(batch)
            rep = get_pool_emb(hidden_state1, pooler, batch[0].gt(0).long(), 'first-last-avg')
            reps.extend(rep.cpu().numpy().tolist())
        return reps


if __name__ == '__main__':
    bc = BertClient(use_tqdm=True)
    bc.encode(['科学技术是第一生产力', '我爱北京天安门'])