# instruct_gpt复现

## 说明
- 按照三个步骤复现rlhf的实现
- 仅做小模型上的思路复现
- [数据链接](https://github.com/shibing624/MedicalGPT/tree/main/data)


训练大模型，实现包括二次预训练、有监督微调、奖励建模、强化学习训练。

<img src="https://github.com/Tongjilibo/bert4torch/blob/master/docs/pics/GPT_Training.jpg" width="860" />

分四阶段训练GPT模型，来自Andrej Karpathy的演讲PDF [State of GPT](https://karpathy.ai/stateofgpt.pdf)，视频 [Video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)


## 参考repo
- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)