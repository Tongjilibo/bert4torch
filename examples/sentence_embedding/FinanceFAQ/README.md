# 召回+排序两阶段模型
金融场景FAQ解决方案

## 建模思路
1. 阶段一：MultiNegativeRankingLoss来做有监督的语义相似度任务
2. 利用1阶段训练好的模型，用向量相似度为所有的相似问q_sim召回最相近的K个标准问q_std_pred, 其中等于q_std的为正样本，不等于的为困难负样本
3. 阶段二：ContrastiveLoss来做有监督的语义相似度任务
4. 预测：一个query通过阶段一模型找到topK个标问q_std, 然后通过阶段二模型从topK个标问中找到最可能的标问

## 优缺点分析
- 阶段一的训练自动为阶段二模型构造困难样本，类似于Boosting的思想，进一步提升准确率

## 文件说明
| 文件名 | 文件描述 |
| ----  |  ----  |
| task_sentence_embedding_FinanceFAQ_step1_0.ipynb | 阶段一模型数据生成 |
| task_sentence_embedding_FinanceFAQ_step1_1.py | 阶段一模型训练 |
| task_sentence_embedding_FinanceFAQ_step2_0.ipynb | 阶段二模型数据生成 |
| task_sentence_embedding_FinanceFAQ_step2_1.ipynb | 阶段二模型训练 |
| task_sentence_embedding_FinanceFAQ_step3_predict.ipynb | 模型效果评估 |
| task_sentence_embedding_FinanceFAQ_step3_inference.ipynb | 单条样本推理 |

## 指标
- 评测数据集：所有标问相似问pair（样本内）
- 指标：recall（正确标问在召回的TopK中的比例）

| 阶段 | Top1 | Top3 | Top5 | Top10 |
|----|----|----|----|----|
|一阶段raw方式|91.32|97.94|98.91|99.57|
|一阶段random方式|88.19|95.93|97.56|98.82|
|一阶段mul_ce方式|90.32|97.51|98.67|99.44|
|二阶段|98.00|99.47|99.79|100|
|一阶段raw方式+二阶段整体|97.54|99.00|99.33|99.50|

## requirements
transformers==4.15.0