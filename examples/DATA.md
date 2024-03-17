# 一、数据集

## 传统nlp任务
| 数据集名称     | 用途               | 备注                                       |
| ---------------- | -------------------- | ------------------------------------------ |
| 人民日报数据集 | 实体识别           | [china-people-daily-ner-corpus](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)                               |
| 百度关系抽取   | 关系抽取           | [官网](http://ai.baidu.com/broad/download?dataset=sked), [百度云(含dev)](https://pan.baidu.com/s/1aWXDkJkiMegzvwZ1XsuO2Q?pwd=5945), [HF]()|
| Sentiment      | 情感分类           | [Sentiment](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)                                   |
| THUCNews       | 文本分类、文本生成 | [源文件](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews), [HF(转换后)](https://huggingface.co/datasets/Tongjilibo/THUCNews) |
| ATEC           | 文本相似度         | [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)                                                           |
| BQ             | 文本相似度         | [BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)                                                                               |
| LCQMC          | 文本相似度         | [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)                                                                         |
| PAWSX          | 文本相似度         | [PAWSX](https://arxiv.org/abs/1908.11828)                                                                                       |
| STS-B          | 文本相似度         | [STS-B](https://github.com/pluto-junzeng/CNSD)                                                                                  |
| CSL            | 文本生成           | [CSL](https://github.com/CLUEbenchmark/CLGE)                                                                                    |

## 预训练
- Wiki中文百科
- 百度百科
- C4_ZH
- [WuDaoCorpora Text文本预训练数据集](https://data.baai.ac.cn/details/WuDaoCorporaText)


## 指令微调
| 数据集名称     | 介绍               |
| ---------------- | -------------------- |
|[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)|参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条|
|[BelleGroup/Belle-0.5M-cn](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)|包含约50万条由BELLE项目生成的中文指令数据||
|[BelleGroup/Belle-1M-cn](https://huggingface.co/datasets/BelleGroup/train_1M_CN)| 包含约100万条由BELLE项目生成的中文指令数据|
|[BelleGroup/Belle-school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)| Belle开放的0.25M数学指令数据集|
|[BelleGroup/Belle-multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)| Belle开放的0.8M多轮任务对话数据集|
|[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)|MOSS-002所使用的多轮对话数据，覆盖有用性、忠实性、无害性三个层面，包含由text-davinci-003生成的约57万条英文对话和59万条中文对话|
|[fnlp/moss-003-sft-data](https://huggingface.co/datasets/fnlp/moss-003-sft-data)|moss-moon-003-sft所使用的多轮对话数据，基于MOSS-002内测阶段采集的约10万用户输入数据和gpt-3.5-turbo构造而成，相比moss-002-sft-data，moss-003-sft-data更加符合真实用户意图分布，包含更细粒度的有用性类别标记、更广泛的无害性数据和更长对话轮数，约含110万条对话数据|
|[YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)|流萤23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万|
|[YeungNLP/ultrachat](https://huggingface.co/datasets/YeungNLP/ultrachat)|清华大学开源的英文多轮对话数据，包含140万+数据|
|[YeungNLP/WizardLM_evol_instruct_V2_143k](https://huggingface.co/datasets/YeungNLP/WizardLM_evol_instruct_V2_143k) | 由WizardLM项目开源的英文指令微调数据集，通过Evol-Instruct方法让指令进化，加强指令的复杂度，以提升模型对复杂指令的遵循能力。包含143k条数据。|
|[shareAI/CodeChat](https://huggingface.co/datasets/shareAI/CodeChat)      | 主要包含逻辑推理、代码问答、代码生成相关语料样本。 |
|[shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)     | 中英文平行双语优质人机问答数据集，覆盖真实复杂场景下的用户提问。|
|[YeungNLP/ultrafeedback_binarized](https://huggingface.co/datasets/YeungNLP/ultrafeedback_binarized)      | 英文偏好数据集，可用于DPO训练   |
|[deepctrl/deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary)|匠数大模型SFT数据集是一个由匠数科技精心搜集整理的高质量数据集,包含10M条数据的中文数据集和包含2M条数据的英文数据集|

# 二、指标测试

## 1. 文本分类

### 1.1 不同预训练模型的指标对比

- [情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls位分类


| solution     | epoch       | valid_acc         | test_acc          | comment                                                    |
| -------------- | ------------- | ------------------- | ------------------- | ------------------------------------------------------------ |
| albert_small | 10/10       | 94.46             | 93.98             | small版本                                                  |
| bert         | 6/10        | 94.72             | 94.11             | ——                                                       |
| robert       | 4/10        | 94.77             | 94.64             | ——                                                       |
| nezha        | 7/10        | 95.07             | 94.72             | ——                                                       |
| xlnet        | 6/10        | 95.07             | 94.77             | ——                                                       |
| electra      | 10/10       | 94.94             | 94.78             | ——                                                       |
| roformer     | 9/10        | 94.85             | 94.42             | ——                                                       |
| roformer_v2  | 3/10        | 95.78             | 96.09             | ——                                                       |
| gau_alpha    | 2/10        | 95.25             | 94.46             | ——                                                       |
| deberta_v2   | (10/2/5)/10 | 94.55/95.16/94.85 | 94.99/94.68/94.33 | 分别为97M/320M/710M, 97M的transformer版本指标为94.90/94.81 |

### 1.2 不同trick下的指标对比

- trick测试+[情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls分类+无segment_input


| solution            | epoch | valid_acc | test_acc | comment |
| --------------------- | ------- | ----------- | ---------- | --------- |
| bert                | 10/10 | 94.90     | 94.78    | ——    |
| fgm                 | 4/10  | 95.34     | 94.99    | ——    |
| pgd                 | 6/10  | 95.34     | 94.64    | ——    |
| gradient_penalty    | 7/10  | 95.07     | 94.81    | ——    |
| vat                 | 8/10  | 95.21     | 95.03    | ——    |
| ema                 | 7/10  | 95.21     | 94.86    | ——    |
| ema+warmup          | 7/10  | 95.51     | 95.12    | ——    |
| mix_up              | 6/10  | 95.12     | 94.42    | ——    |
| R-drop              | 9/10  | 95.25     | 94.94    | ——    |
| UDA                 | 8/10  | 94.90     | 95.56    | ——    |
| semi-vat            | 10/10 | 95.34     | 95.38    | ——    |
| temporal_ensembling | 8/10  | 94.94     | 94.90    | ——    |

## 2. 序列标注

- [人民日报数据集](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)+bert预训练模型
- valid集指标


| solution                      | epoch | f1_token | f1_entity | comment                                                      |
| ------------------------------- | ------- | ---------- | ----------- | -------------------------------------------------------------- |
| bert+crf                      | 18/20 | 96.89    | 96.05     | ——                                                         |
| bert+crf+init                 | 18/20 | 96.93    | 96.08     | 用训练数据初始化crf权重                                      |
| bert+crf+freeze               | 11/20 | 96.89    | 96.13     | 用训练数据生成crf权重(不训练)                                |
| bert+cascade+crf              | 5/20  | 98.10    | 96.26     | crf类别少所以f1_token偏高                                    |
| bert+crf+posseg               | 13/20 | 97.32    | 96.55     | 加了词性输入                                                 |
| bert+global_pointer           | 18/20 | ——     | 95.66     | ——                                                         |
| bert+efficient_global_pointer | 17/20 | ——     | 96.55     | ——                                                         |
| bert+mrc                      | 7/20  | ——     | 95.75     | ——                                                         |
| bert+span                     | 13/20 | ——     | 96.31     | ——                                                         |
| bert+tplinker_plus            | 20/20 | ——     | 95.71     | 长度限制明显                                                 |
| uie                           | 20/20 | ——     | 96.57     | zeroshot:f1=60.8, fewshot-100样本:f1=85.82, 200样本:f1=86.40 |
| W2NER                         | 18/20 | 97.37    | 96.32     | 对显存要求较高                                               |
| CNN_Nested_NER                | 19/20 | 98.06    | 96.11     |                                                              |
| LEAR                          | 8/20  | ——     | 96.52     |                                                              |

## 3. 文本表示

### 3.1 无监督语义相似度

- bert预训练模型 + 无监督finetune + cls位句向量(PromptBert除外)
- 五个中文数据集 + 5个epoch取最优值 + valid的spearmanr相关系数
- 继续finetune, 部分数据集有小幅提升
- 实验显示dropout_rate对结果影响较大


| solution        | ATEC  | BQ    | LCQMC | PAWSX | STS-B | comment                               |
| ----------------- | ------- | ------- | ------- | ------- | ------- | --------------------------------------- |
| Bert-whitening  | 26.79 | 31.81 | 56.34 | 17.22 | 67.45 | cls+不降维                            |
| CT              | 30.65 | 44.50 | 68.67 | 16.20 | 69.27 | dropout=0.1, 收敛慢跑了10个epoch      |
| CT_In_Batch_Neg | 32.47 | 47.09 | 68.56 | 27.50 | 74.00 | dropout=0.1                           |
| TSDAE           | ——  | 46.65 | 65.30 | 12.54 | ——  | dropout=0.1, ——表示该指标异常未记录 |
| SimCSE          | 33.90 | 50.29 | 71.81 | 13.14 | 71.09 | dropout=0.3                           |
| ESimCSE         | 34.05 | 50.54 | 71.58 | 12.53 | 71.27 | dropout=0.3                           |
| DiffSCE         | 33.04 | 48.17 | 71.51 | 12.91 | 71.10 | dropout=0.3, 没啥效果                 |
| PromptBert      | 33.98 | 49.89 | 73.18 | 13.30 | 73.42 | dropout=0.3                           |

### 3.2 有监督语义相似度

- bert预训练模型 + 训练数据finetune + cls位句向量
- 五个中文数据集 + 5个epoch取最优值 + valid/test的spearmanr相关系数
- STS-B任务是5分类，其余是2分类


| solution            | ATEC          | BQ            | LCQMC         | PAWSX         | STS-B         | comment          |
| --------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ------------------ |
| CoSENT              | 50.61 / 49.81 | 72.84 / 71.61 | 77.79 / 78.74 | 55.00 / 56.00 | 83.48 / 80.06 |                  |
| ContrastiveLoss     | 50.02 / 49.19 | 72.52 / 70.98 | 77.49 / 78.27 | 58.21 / 57.65 | 69.87 / 68.58 | STS-B转为2分类   |
| InfoNCE             | 47.77 / 46.99 | 69.86 / 68.14 | 71.74 / 74.54 | 52.82 / 54.21 | 83.31 / 78.72 | STS-B转为2分类   |
| concat CrossEntropy | 48.71 / 47.62 | 72.16 / 70.07 | 78.44 / 78.77 | 51.46 / 52.28 | 61.31 / 56.62 | STS-B转为2分类   |
| CosineMSELoss       | 46.89 / 45.86 | 72.27 / 71.35 | 75.29 / 77.19 | 54.92 / 54.35 | 81.64 / 77.76 | STS-B标准化到0-1 |

## 4. 关系提取

- [百度关系提取数据集-官网](http://ai.baidu.com/broad/download?dataset=sked), [百度云(含dev)](https://pan.baidu.com/s/1aWXDkJkiMegzvwZ1XsuO2Q?pwd=5945)


| solution      | f1    | comment                |
| --------------- | ------- | ------------------------ |
| CasRel        | 81.87 |                        |
| gplinker      | 82.38 |                        |
| tplinker      | 74.49 | seq_len=64, 未完全收敛 |
| tplinker_plus | 79.30 | seq_len=64             |
| SPN4RE        | 77.53 |                        |
| PRGC          | 80.36 | 训练很慢               |

## 5. 文本生成

- [CSL数据集](https://github.com/CLUEbenchmark/CLGE)，注意是训练集1万左右的版本，分别dev/test指标


| solution   | Rouge-L       | Rouge-1       | Rouge-2       | BLEU          | comment |
| ------------ | --------------- | --------------- | --------------- | --------------- | --------- |
| bert+unlim | 63.65 / 63.01 | 66.25 / 66.34 | 54.48 / 54.81 | 44.21 / 44.60 |         |
| bart       | 64.62 / 64.99 | 67.72 / 68.40 | 56.08 / 57.26 | 46.15 / 47.67 |         |
| mt5        | 67.67 / 65.98 | 70.39 / 69.36 | 59.60 / 59.05 | 50.34 / 50.11 |         |
| t5_pegasus | 66.07 / 66.11 | 68.94 / 69.61 | 57.12 / 58.38 | 46.14 / 47.95 |         |
| uer_t5     | 63.59 / 63.11 | 66.56 / 66.48 | 54.65 / 54.82 | 44.27 / 44.60 |         |


## 6. 大模型指令微调

- [ADGEN数据集](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)(广告生成)

|            chatglm              |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
| ----------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
| hf+pt2 official+v100-int4-bs1   |   ——      |      ——      |     24.97     |    31.12    |     7.11    |    8.10   |         |
| hf+pt2 reappear+v100-int4-bs1   |   ——      |      ——      |     24.80     |    30.97    |     6.98    |    7.85   |         |
| b4t+pt2+v100+int4+bs1           |   ——      |      ——      |     24.58     |    30.76    |     7.12    |    8.12   |         |
| b4t+pt2+T4-int8-bs1             |  10G      |     1470     |     24.87     |    30.83    |     7.14    |    8.05   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs1 |  15G      |     287      |     25.10     |    31.43    |     7.30    |    8.28   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs8 |  22G      |     705      |     25.22     |    31.22    |     7.38    |    8.35   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs1 |  29G      |     760      |     24.83     |    30.95    |     7.18    |    8.08   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs4 |  32G      |     2600     |     25.12     |    31.55    |     7.21    |    8.02   |         |
| b4t+lora+V100-fp16-bs16         |  28G      |     2570     |     24.89     |    31.38    |     7.17    |    8.15   |         |
| b4t+qlora+V100-bs16             |  26G      |     5381     |     23.99     |    29.52    |     6.47    |    7.74   |         |
