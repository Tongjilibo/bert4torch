# 1. 文本分类
## 1.1 不同预训练模型的指标对比
- [情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls位分类

| solution | epoch | valid_acc | test_acc | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| albert_small | 10/10 | 94.46 | 93.98 | small版本 | 
| bert | 6/10 | 94.72 | 94.11 | —— | 
| robert | 4/10 | 94.77 | 94.64 | —— | 
| nezha | 7/10 | 95.07 | 94.72 | —— | 
| xlnet | 6/10 | 95.00 | 94.24 | —— | 
| electra | 10/10 | 94.94 | 94.78 | —— | 
| roformer | 9/10 | 94.85 | 94.42 | —— | 
| roformer_v2 | 3/10 | 95.78 | 96.09 | —— | 
| gau_alpha | 2/10 | 95.25 | 94.46 | —— | 

## 1.2 不同trick下的指标对比
- trick测试+[情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls分类+无segment_input

| solution | epoch | valid_acc | test_acc | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| bert | 10/10 | 94.90 | 94.78 | —— | 
| fgm | 4/10 | 95.34 | 94.99 | —— | 
| pgd | 6/10 | 95.34 | 94.64 | —— | 
| gradient_penalty | 7/10 | 95.07 | 94.81 | —— | 
| vat | 8/10 | 95.21 | 95.03 | —— | 
| ema | 7/10 | 95.21 | 94.86 | —— | 
| ema+warmup | 7/10 | 95.51 | 95.12 | —— | 
| mix_up | 6/10 | 95.12 | 94.42 | —— | 
| R-drop | 9/10 | 95.25 | 94.94 | —— | 
| UDA | 8/10 | 94.90 | 95.56 | —— | 
| semi-vat | 10/10 | 95.34 | 95.38 | —— |
| temporal_ensembling | 8/10 | 94.94 | 94.90 | —— |

# 2. 序列标注
- [人民日报数据集](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)+bert预训练模型
- valid集指标

| solution | epoch | f1_token | f1_entity | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| bert+crf | 18/20 | 96.89 | 96.05 | —— |
| bert+crf+init | 18/20 | 96.93 | 96.08 | 用训练数据初始化crf权重 | 
| bert+crf+freeze | 11/20 | 96.89 | 96.13 | 用训练数据生成crf权重(不训练) |
| bert+cascade+crf | 5/20 | 98.10 | 96.26 | crf类别少所以f1_token偏高 | 
| bert+crf+posseg | 13/20 | 97.32 | 96.55 | 加了词性输入 | 
| bert+global_pointer | 18/20 | —— | 95.66 | —— | 
| bert+efficient_global_pointer | 17/20 | —— | 96.55 | —— | 
| bert+mrc | 7/20 | —— | 95.75 | —— |
| bert+span | 13/20 | —— | 96.31 | —— |
| bert+tplinker_plus | 20/20 | —— | 95.71 | 长度限制明显 |
| uie | 20/20 | —— | 96.57 | zeroshot:f1=60.8, fewshot-100样本:f1=85.82, 200样本:f1=86.40 |
| W2NER | 18/20 | 97.37 | 96.32 | 对显存要求较高 |
| CNN_Nested_NER |19/20|98.06|96.11|  |

# 3. 文本表示
## 3.1 无监督语义相似度
- bert预训练模型 + 无监督finetune + cls位句向量(PromptBert除外)
- 五个中文数据集 + 5个epoch取最优值 + valid的spearmanr相关系数
- 继续finetune, 部分数据集有小幅提升
- 实验显示dropout_rate对结果影响较大

|     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |   comment   |
|       ----      |   ----  | ---- |   ----  |   ----  |   ----  |     ----    |
| Bert-whitening  |  26.79  | 31.81|  56.34  |  17.22  |  67.45  | cls+不降维   |
|        CT       |  30.65  | 44.50|  68.67  |  16.20  |  69.27  | dropout=0.1, 收敛慢跑了10个epoch |
| CT_In_Batch_Neg |  32.47  | 47.09|  68.56  |  27.50  |  74.00  | dropout=0.1 |
|       TSDAE     |    ——   | 46.65|  65.30  |  12.54  |    ——   | dropout=0.1, ——表示该指标异常未记录 |
|      SimCSE     |  33.90  | 50.29|  71.81  |  13.14  |  71.09  | dropout=0.3 |
|      ESimCSE    |  34.05  | 50.54|  71.58  |  12.53  |  71.27  | dropout=0.3 |
|      DiffSCE    |  33.04  | 48.17|  71.51  |  12.91  |  71.10  | dropout=0.3, 没啥效果 |
|    PromptBert   |  33.98  | 49.89|  73.18  |  13.30  |  73.42  | dropout=0.3 |

## 3.2 有监督语义相似度
- bert预训练模型 + 训练数据finetune + cls位句向量
- 五个中文数据集 + 5个epoch取最优值 + valid/test的spearmanr相关系数
- STS-B任务是5分类，其余是2分类

|      solution     |     ATEC    |     BQ      |    LCQMC    |    PAWSX    |    STS-B    |      comment     |
|        ----       |     ----    |    ----     |     ----    |     ----    |     ----    |        ----      |
|       CoSENT      |50.61 / 49.81|72.84 / 71.61|77.79 / 78.74|55.00 / 56.00|83.48 / 80.06|                  |
|  ContrastiveLoss  |50.02 / 49.19|72.52 / 70.98|77.49 / 78.27|58.21 / 57.65|69.87 / 68.58|   STS-B转为2分类  |
|      InfoNCE      |47.77 / 46.99|69.86 / 68.14|71.74 / 74.54|52.82 / 54.21|83.31 / 78.72|   STS-B转为2分类  |
|concat CrossEntropy|48.71 / 47.62|72.16 / 70.07|78.44 / 78.77|51.46 / 52.28|61.31 / 56.62|   STS-B转为2分类  |
|   CosineMSELoss   |46.89 / 45.86|72.27 / 71.35|75.29 / 77.19|54.92 / 54.35|81.64 / 77.76|  STS-B标准化到0-1 |

# 4. 关系提取
- [百度关系提取数据集](http://ai.baidu.com/broad/download?dataset=sked)

|      solution     |     f1    |    comment  |
|        ----       |    ----   |    ----     |
|       CasRel      |   81.87   |             |
|     gplinker      |   81.88   |             |
|     tplinker      |   74.49   |  seq_len=64, 未完全收敛 |
|    tplinker_plus  |   79.30   |  seq_len=64 |

# 5. 文本生成
- [CSL数据集](https://github.com/CLUEbenchmark/CLGE)，注意是训练集1万左右的版本，分别dev/test指标

| solution |   Rouge-L   |   Rouge-1   |   Rouge-2   |     BLEU    |    comment  |
|   ----   |     ----    |     ----    |     ----    |     ----    |    ----     |
|bert+unlim|63.65 / 63.01|66.25 / 66.34|54.48 / 54.81|44.21 / 44.60|             |
|   bart   |64.62 / 64.99|67.72 / 68.40|56.08 / 57.26|46.15 / 47.67|             |
|   mt5    |67.67 / 65.98|70.39 / 69.36|59.60 / 59.05|50.34 / 50.11|             |
|t5_pegasus|66.07 / 66.11|68.94 / 69.61|57.12 / 58.38|46.14 / 47.95|             |
|  uer_t5  |63.59 / 63.11|66.56 / 66.48|54.65 / 54.82|44.27 / 44.60|             |