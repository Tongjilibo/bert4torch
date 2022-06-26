# sequence_labeling
- 人民日报数据集+bert预训练模型
- valid集指标

| solution | epoch | f1_token | f1_entity | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| bert+crf | 6/20 | 97.02 | 96.22 | —— | 
| bert+cascade+crf | 15/20 | 98.11 | 96.23 | crf类别少所以f1_token偏高 | 
| bert+posseg+crf | 8/20 | 97.30 | 96.09 | —— | 
| bert+global_pointer | 18/20 | —— | 95.66 | —— | 
| bert+efficient_global_pointer | 17/20 | —— | 96.55 | —— | 
| bert+mrc | 7/20 | —— | 95.75 | —— |
| bert+span | 13/20 | —— | 96.31 | —— |
| bert+tplinker_plus | 20/20 | —— | 95.71 | 长度限制明显 |


# sentence_embedding
## unsupervised
- bert预训练模型+无监督finetune
- 五个中文数据集+仅跑了一个epoch
- 继续finetune, 部分数据集有小幅提升

| solution |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
|   ----   |   ----  | ---- |   ----  |   ----  |   ----  |
|  SimCSE  |  33.30  | 49.95|  70.36  |  12.69  |  69.00  |
|  ESimCSE |  33.61  | 50.43|  70.61  |  12.84  |  69.31  |
|PromptBert|  33.61  | 47.54|  71.81  |  24.47  |  73.97  |

## supervised
待整理

# sentence_classfication
- 情感分类数据集+cls位分类

| solution | epoch | valid_acc | test_acc | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| albert_small | 10/10 | 94.46 | 93.98 | small版本 | 
| bert | 6/10 | 94.72 | 94.11 | —— | 
| nezha | 7/10 | 95.07 | 94.72 | —— | 
| xlnet | 6/10 | 95.00 | 94.24 | —— | 
| electra | 10/10 | 94.94 | 94.78 | —— | 
| roformer | 9/10 | 94.85 | 94.42 | —— | 
| roformer_v2 | 3/10 | 95.78 | 96.09 | —— | 
| gau_alpha | 2/10 | 95.25 | 94.46 | —— | 