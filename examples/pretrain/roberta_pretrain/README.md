# 使用说明
## 思路
RoBERTa相比于BERT的更改有一项是动态mask，各自的实现方式如下
- BERT: 将语料进行mask操作后存储下来，模型在每一个epoch针对同一份数据重复学习。
- RoBERTa: 将数据复制10份，分别进行不同的随机 mask，喂给模型进行学习。文中说是动态mask，其实还是静态的。

我这里实现的方式是`数据生成`和`模型训练`同时进行，`数据生成`读入txt文件并转换成待训练文件，`模型训练`随机选择一份待训练模型进行训练，训练完即销毁，这样使得`数据生成`对同一份txt文件先后会使用不同的mask方式来生成训练文件，来实现动态mask

## 使用方式
1. 先运行起来`pretrain_roberta_mlm_data_gen.py`（一直运行，一直在生成数据），其中`dir_corpus`是读入的txt语料地址，`dir_training_data`是保存的文件目录（生成的待训练数据地址）
2. 运行`pretrain_roberta_mlm.py`，其中`dir_training_data`即为`pretrain_roberta_mlm_data_gen.py`中的`dir_training_data`
3. 训练过程中，两个进程一直跑着，一般我会一个terminal跑数据生成，另一个termimal跑模型训练