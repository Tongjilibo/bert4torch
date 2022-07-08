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