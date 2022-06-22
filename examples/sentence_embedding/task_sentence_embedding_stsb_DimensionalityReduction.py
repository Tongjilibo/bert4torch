#! -*- coding:utf-8 -*-
# 利用pca压缩句向量
# 从768维压缩到128维，指标从81.82下降到80.10

from task_sentence_embedding_stsb_CosineSimilarityLoss import model, train_dataloader, Model, device, valid_dataloader, evaluate
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn

new_dimension = 128  # 压缩到的维度

train_embeddings = []
for token_ids_list, labels in train_dataloader:
    for token_ids in token_ids_list:
        train_embeddings.append(model.encode(token_ids))
    # if len(train_embeddings) >= 20:
    #     break
train_embeddings = torch.concat(train_embeddings, dim=0).cpu().numpy()
print('train_embeddings done, start pca training...')

pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)
print('PCA training done...')

# 定义bert上的模型结构
class NewModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Linear(768, new_dimension, bias=False)
        self.dense.weight = torch.nn.Parameter(torch.tensor(pca_comp, device=device))

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask)
            output = self.dense(output)
        return output

new_model = NewModel().to(device)
new_model.load_weights('best_model.pt', strict=False)

print('Start evaludating...')
val_consine = evaluate(new_model, valid_dataloader)
print(f'val_consine: {val_consine:.5f}\n')