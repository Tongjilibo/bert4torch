# 模型推理脚本
# cv逐一预测，按照dev的指标加权
from copyreg import pickle
from torch import device
from training import Model, collate_fn
import torch
from torch.utils.data import DataLoader
from bert4torch.snippets import ListDataset
import pandas as pd
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
def load_data(df):
    """加载数据。"""
    D = list()
    for _, row in df.iterrows():
        text = row['text']
        D.append((text, 0))
    return D

df_test = pd.read_csv('E:/Github/天池新闻分类/data/test_a.csv', sep='\t')
df_test['text'] = df_test['text'].apply(lambda x: x.strip().split())
test_data = load_data(df_test)
dev_dataloader = DataLoader(ListDataset(data=test_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

f1_score = [0.97083, 0.97074, 0.96914, 0.96892, 0.96613]
y_pred_final = 0
for i in range(5):
    model = Model().to(device)
    model.load_weights(f'best_model_fold{i+1}.pt')
    y_pred = []
    for x, _ in tqdm(dev_dataloader, desc=f'evaluate_cv{i}'):
        y_pred.append(model.predict(x).cpu().numpy())
        # if len(y_pred) > 10:
        #     break
    y_pred = np.concatenate(y_pred)
    y_pred_final += y_pred * f1_score[i]
    np.save(f'test_cv{i}_logit.npy', y_pred)

df_test = pd.DataFrame(y_pred_final.argmax(axis=1))
df_test.columns = ['label']
df_test.to_csv('submission.csv', index=False)