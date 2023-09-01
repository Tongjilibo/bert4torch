# instruct_gpt复现

- 按照三个步骤复现rlhf的实现
- 仅做小模型上的思路复现
- [数据链接](https://github.com/shibing624/MedicalGPT/tree/main/data)


训练大模型，实现包括二次预训练、有监督微调、奖励建模、强化学习训练。

<img src="https://github.com/Tongjilibo/bert4torch/blob/master/docs/pics/GPT_Training.jpg" width="860" />

分四阶段训练GPT模型，来自Andrej Karpathy的演讲PDF [State of GPT](https://karpathy.ai/stateofgpt.pdf)，视频 [Video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)


## 文件说明
0. `step0_continue_pretrain.py`: 继续预训练
1. `step1_sft.py`: 指令微调
2. `step2_reward.py`: 奖励模型
3. `step3_rlhf.py`: 基于人类反馈的强化学习
4. `step4_dpo.py`: dpo算法，可用于替代 "奖励模型+rlhf" 两个步骤
5. `utils.py`: 功能函数


## 参考repo
- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)