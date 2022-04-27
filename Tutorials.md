# bert4torch使用教程

## 1. 建模流程示例
```python
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集，可以自己继承Dataset来定义
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """读取文本文件，整理成需要的格式
        """
        D = []
        return D

def collate_fn(batch):
    '''处理上述load_data得到的batch数据，整理成对应device上的Tensor
    注意：返回值分为feature和label, feature可整理成list或tuple
    '''
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset('file_path'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 定义bert上的模型结构，以文本二分类为例
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 2)

    def forward(self, token_ids, segment_ids):
        # build_transformer_model得到的模型仅接受list/tuple传参，因此入参只有一个时候包装成[token_ids]
        hidden_states, pooled_output = self.bert([token_ids, segment_ids])
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(), # 可以自定义Loss
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 可以自定义优化器
    scheduler=None, # 可以自定义scheduler
    metrics=['accuracy']
)

# 定义评价函数
def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    return right / total

class Evaluator(Callback):
    """评估与保存，这里定义仅在epoch结束后调用
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=100, grad_accumulation_steps=2, callbacks=[evaluator])
```

## 2. 主要模块讲解
### 1) 数据处理部分
#### a. 精简词表，并建立优化器
```python
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,  # 词典文件路径
    simplified=True,  # 过滤冗余部分token，如[unused1]
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],  # 指定起始的token，如[UNK]从bert默认的103位置调整到1
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 若无需精简，仅使用当前行定义tokenizer即可
```
### b. 好用的小函数
- `text_segmentate()`: 截断总长度至不超过maxlen, 接受多个sequence输入，每次截断最长的句子，indices表示删除的token位置
- `tokenizer.encode()`: 把text转成token_ids，默认句首添加[CLS]，句尾添加[SEP]，返回token_ids和segment_ids，相当于同时调用`tokenizer.tokenize()`和`tokenizer.tokens_to_ids()`
- `tokenizer.decode()`: 把token_ids转成text，默认会删除[CLS], [SEP], [UNK]等特殊字符，相当于调用`tokenizer.ids_to_tokens()`并做了一些后处理
- `sequence_padding`: 将序列padding到同一长度, 传入一个元素为list, ndarray, tensor的list，返回ndarry或tensor


### 2) 模型定义部分
- 模型创建
```python
'''
调用模型后，若设置with_pool, with_nsp, with_mlm，则返回值依次为[hidden_states, pool_emb/nsp_emb, mlm_scores],否则只返回hidden_states
'''
build_transformer_model(
    config_path=config_path, # 模型的config文件地址
    checkpoint_path=checkpoint_path, # 模型文件地址，默认值None表示不加载预训练模型
    model='bert', # 加载的模型结构，这里Model也可以基于nn.Module自定义后传入
    application='encoder',  # 模型应用，支持encoder，lm和unilm格式
    segment_vocab_size=2,  # type_token_ids数量，默认为2，如不传入segment_ids则需设置为0
    with_pool=False,  # 是否包含Pool部分
    with_nsp=False,  # 是否包含NSP部分
    with_mlm=False,  # 是否包含MLM部分
    return_model_config=False,  # 是否返回模型配置参数
    output_all_encoded_layers=False,  # 是否返回所有hidden_state层
)
```

- 定义loss，optimizer，scheduler等
```python
'''
定义使用的loss和optimizer，这里支持自定义
'''
model.compile(
    loss=nn.CrossEntropyLoss(), # 可以自定义Loss
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 可以自定义优化器
    scheduler=None, # 可以自定义scheduler
    adversarial_train={'name': 'fgm'},  # 训练trick方案设置，支持fgm, pgd, gradient_penalty, vat
    metrics=['accuracy']  # loss等默认打印的字段无需设置
)
```

- 自定义模型
```python
'''
基于bert上层的各类魔改，如last2layer_average, token_first_last_average
'''
class Model(BaseModel):
    # 需要继承BaseModel
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path)
    def forward(self):
        pass
```

- [自定义训练过程](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_custom_fit_progress.py)
```python
'''
自定义fit过程，适用于自带fit()不满足需求时，如混合精度等
'''
class Model(BaseModel):
    def fit(self, train_dataloader, steps_per_epoch, epochs):   
        train_dataloader = cycle(train_dataloader)
        self.train()
        for epoch in range(epochs):
            for bti in range(steps_per_epoch):
                train_X, train_y = next(train_dataloader)
                output = self.forward(*train_X)
                loss = self.criterion(output, train_y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
```

### 3) 模型评估部分
```python
'''支持在多个位置执行
'''
class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.
    def on_train_begin(self, logs=None):  # 训练开始时候
        pass
    def on_train_end(self, logs=None):  # 训练结束时候
        pass
    def on_batch_begin(self, global_step, batch, logs=None):  # batch开始时候
        pass
    def on_batch_end(self, global_step, batch, logs=None):  # batch结束时候
        # 可以设置每隔多少个step，后台记录log，写tensorboard等
        # 尽量不要在batch_begin和batch_end中print，防止打断进度条功能
        pass
    def on_epoch_begin(self, global_step, epoch, logs=None):  # epoch开始时候
        pass
    def on_epoch_end(self, global_step, epoch, logs=None):  # epoch结束时候
        val_acc = evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')
```
## 3. 其他特性讲解
### 1) tensorboard保存训练过程
```python
from tensorboardX import SummaryWriter
class Evaluator(Callback):
    """每隔多少个step评估并记录tensorboard
    """
    def on_batch_end(self, global_step, batch, logs=None):
        if global_step % 100 == 0:
            writer.add_scalar(f"train/loss", logs['loss'], global_step)
            val_acc = evaluate(valid_dataloader)
            writer.add_scalar(f"valid/acc", val_acc, global_step)

```
### 2) 打印训练参数
```python
from torchinfo import summary
summary(model, input_data=next(iter(train_dataloader))[0])
```