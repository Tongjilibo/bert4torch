#! -*- coding: utf-8 -*-
# 调用PromptCLUE https://github.com/clue-ai/PromptCLUE
# huggingface: https://huggingface.co/ClueAI/PromptCLUE-base-v1-5

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.generation import AutoRegressiveDecoder


# 配置
pretrain_model = 'E:/data/pretrain_ckpt/ClueAI/PromptCLUE-base-v1-5/'
config_path = pretrain_model + 'bert4torch_config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
spm_path = pretrain_model + 'spiece.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)

model = build_transformer_model(config_path, checkpoint_path, pad_token_id=-1).to(device)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        res = model.decoder.predict([output_ids] + inputs)
        return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

    def generate(self, text, n=1, top_p=1, temperature=0.7):
        text = text.replace("\n", "_")
        token_ids, _ = tokenizer.encode(text, maxlen=768)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.random_sample(encoder_output, n=n, top_p=top_p, temperature=temperature)  # 基于随机采样
        out_text = tokenizer.decode([int(i.cpu().numpy()) for i in output_ids[0]])
        return out_text.replace("_", "\n")

autotitle = AutoTitle(bos_token_id=0, eos_token_id=tokenizer._token_end_id, max_new_tokens=512, device=device)

if __name__ == '__main__':
    input_text0 = "生成与下列文字相同意思的句子： 白云遍地无人扫 答案："
    input_text1 = "用另外的话复述下面的文字： 怎么到至今还不回来，这满地的白云幸好没人打扫。 答案："
    input_text2 = "改写下面的文字，确保意思相同： 一个如此藐视本国人民民主权利的人，怎么可能捍卫外国人的民权？ 答案："
    input_text3 = "根据问题给出答案： 问题：手指发麻的主要可能病因是： 答案："
    input_text4 = "问题：黄果悬钩子的目是： 答案："
    input_list = [input_text0, input_text1, input_text2, input_text3, input_text4]
    for i, input_text in enumerate(input_list):
        print(f"示例{i}".center(50, "="))
        output_text = autotitle.generate(input_text)
        print(f"{input_text}{output_text}")