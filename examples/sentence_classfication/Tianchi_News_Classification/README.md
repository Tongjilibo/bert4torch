# 天池新闻分类
比赛链接：https://tianchi.aliyun.com/competition/entrance/531810/introduction?lang=zh-cn

| 解决方案 | 说明 | 指标 |
| ---- | ---- | ---- |
| Top1 | [Github](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION)| 正式赛f1=0.9735 |
| Top1复现 | bert以第1折交叉epoch=5初始化，1个epoch，seed=0, 1993, 2020三者融合 | 长期赛f1=0.9736 |
| Top1_bert4torch复现 | bert+attn+fgm+cv | 长期赛f1=0.9727, dev_5cv=(0.97083, 0.97074, 0.96914, 0.96892, 0.96613)|

## 文件说明
- convert.py: 将上述链接中的tensorflow权重转为pytorch的
- training.py: finetune训练代码