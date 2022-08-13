#! -*- coding:utf-8 -*-
# 模型压缩，仅保留bert-base部分层
# 初测测试指标从80%降到77%左右，未细测

from task_sentence_embedding_sup_CosineMSELoss import model, train_dataloader, Model, device, valid_dataloader, evaluate
from bert4torch.snippets import Callback, get_pool_emb
import torch.optim as optim
import torch.nn as nn
from bert4torch.models import build_transformer_model


train_token_ids, train_embeddings = [], []
for token_ids_list, labels in train_dataloader:
    train_token_ids.extend(token_ids_list)
    for token_ids in token_ids_list:
        train_embeddings.append(model.encode(token_ids))
    # if len(train_embeddings) >= 20:
    #     break

new_train_dataloader = list(zip(train_token_ids, train_embeddings))
print('train_embeddings done, start model distillation...')


# 仅取固定的层
class NewModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
        self.bert = build_transformer_model(config_path=config_path, with_pool=True, segment_vocab_size=0, keep_hidden_layers=[1,4,7])

    def forward(self, token_ids):
        hidden_state, pooler = self.bert([token_ids])
        attention_mask = token_ids.gt(0).long()
        output = get_pool_emb(hidden_state, pooler, attention_mask, self.pool_method)
        return output

new_model = NewModel().to(device)
new_model.compile(
    loss=nn.MSELoss(),
    optimizer=optim.Adam(new_model.parameters(), lr=2e-5),
)
new_model.load_weights('best_model.pt', strict=False)  # 加载大模型的部分层
val_consine = evaluate(new_model, valid_dataloader)
print('init val_cosine after distillation: ', val_consine)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = evaluate(new_model, valid_dataloader)
        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # new_model.save_weights('best_model.pt')
        print(f'val_consine: {val_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    new_model.fit(new_train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])
else:
    new_model.load_weights('best_model.pt')
