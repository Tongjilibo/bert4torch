#! -*- coding: utf-8 -*-
# 调用chatyuan https://github.com/clue-ai/ChatYuan

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.generation import AutoRegressiveDecoder


# 配置
pretrain_model = 'G:/pretrain_ckpt/t5/[ClueAI_t5_torch_large]--ClueAI-ChatYuan-large-v1/'
config_path = pretrain_model + 'bert4torch_config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
spm_path = pretrain_model + 'spiece.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)

encoder = build_transformer_model(
    config_path,
    checkpoint_path,
    model='mt5.1.1',
    segment_vocab_size=0,
    logit_scale=False,
    pad_token_id=-1,  # 为了不和decoder_start_ids=0冲突
).to(device)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        return encoder.decoder.predict([output_ids] + inputs)[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, n=1, topp=1, temperature=0.7):
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        token_ids, _ = tokenizer.encode(text, maxlen=768)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = encoder.encoder.predict([token_ids])
        output_ids = self.random_sample(encoder_output, n, topp=topp, temperature=temperature)  # 基于随机采样
        out_text = tokenizer.decode([int(i.cpu().numpy()) for i in output_ids[0]])
        return out_text.replace("\\n", "\n").replace("\\t", "\t")

autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=512, device=device)

if __name__ == '__main__':
    input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    input_text1 = "你能干什么"
    input_text2 = "写一封英文商务邮件给英国客户，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
    input_text3 = "写一个文章，题目是未来城市"
    input_text4 = "写一个诗歌，关于冬天"
    input_text5 = "从南京到上海的路线"
    input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
    input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
    input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
    input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6, input_text7, input_text8]
    for i, input_text in enumerate(input_list):
        input_text = "用户：" + input_text + "\n小元："
        print(f"示例{i}".center(50, "="))
        output_text = autotitle.generate(input_text)
        print(f"{input_text}{output_text}")   