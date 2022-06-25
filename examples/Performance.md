# sequence_labeling
- 人民日报数据集+bert预训练模型

| solution | epoch | f1_token | f1_entity | comment | 
| ---- | ---- | ---- | ---- | ---- | 
| bert+crf | 7/20 | 97.01 | 95.53 | loss后续上升提前中止 | 
| bert+cascade+crf | 15/20 | 98.11 | 96.23 | —— | 
| bert+posseg+crf | 3/20 | 96.74 | 94.93 | loss后续上升提前中止 | 
| bert+global_pointer | 18/20 | —— | 95.66 | —— | 
| bert+efficient_global_pointer | 17/20 | —— | 96.55 | —— | 
| bert+mrc | 7/20 | —— | 95.75 | —— |
| bert+span | 13/20 | —— | 96.31 | —— |
| bert+tplinker_plus | 20/20 | —— | 95.71 | 长度限制明显 |


# sentence_embedding
- bert预训练模型+无监督finetune
- 五个中文数据集+仅跑了一个epoch
- 继续finetune, 部分数据集有小幅提升

| solution |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
|   ----   |   ----  | ---- |   ----  |   ----  |   ----  |
|  SimCSE  |  33.30  | 49.95|  70.36  |  12.69  |  69.00  |
|  ESimCSE |  33.61  | 50.43|  70.61  |  12.84  |  69.31  |
|PromptBert|  33.61  | 47.54|  71.81  |  24.47  |  73.97  |

# sentence_classfication
待整理