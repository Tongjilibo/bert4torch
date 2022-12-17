#! -*- coding: utf-8 -*-
# 预训练模型：https://huggingface.co/shibing624/text2vec-base-chinese
# 方案是Cosent方案

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "F:/Projects/pretrain_ckpt/text2vec-base-chinese"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
token_ids, segments_ids = tokenizer.encode(sentences)
tokens_ids_tensor = torch.tensor(sequence_padding(token_ids))
segment_ids_tensor = torch.tensor(sequence_padding(segments_ids))

model.eval()
with torch.no_grad():
    token_embeddings = model([tokens_ids_tensor, segment_ids_tensor])
    attention_mask = (tokens_ids_tensor != tokenizer._token_pad_id).long()
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    print("Sentence embeddings:")
    print(sentence_embeddings)