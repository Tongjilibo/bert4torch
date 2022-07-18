# 搜狐基于实体的情感分类
- 比赛链接：https://www.biendata.xyz/competition/sohu_2022/

| 解决方案 | 链接 | 指标 |
| ---- | ---- | ---- |
| Top1 | [知乎](https://zhuanlan.zhihu.com/p/533808475)| 初赛f1=0.7253, 复赛f1=0.8173 |
| baseline | —— | 初赛f1=0.6737 |

# bert4torch复现
- 由于比赛结束无法提交，复现只使用线下dev作为对比
- dev为前2000，未使用方案中的后10%作为dev, dev指标略微有点不稳定

| 复现方案 | 方案 | 指标 |
| ---- | ---- | ---- |
| Top1_github | 前2000为dev, 不使用swa, 有warmup, 无label_smoothing, 无fgm, 梯度累积=3, 无rdrop | Epoch 4/10: f1=0.7697|
| Top1_bert4torch复现1 | 参数同上 | Epoch 10/10: f1=0.7556 |
| Top1_bert4torch复现2 | 参数同上+fgm+swa | Epoch 5/10: f1=0.7877 |