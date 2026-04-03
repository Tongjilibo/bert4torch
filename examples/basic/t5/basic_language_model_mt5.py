#! -*- coding: utf-8 -*-
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：https://kexue.fm/archives/7867
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
# mt5主要特点：gated-gelu, decoder的最后的dense层独立权重，rmsnorm

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer, load_vocab
from bert4torch.generation import AutoRegressiveDecoder
import torch

# 基本参数
max_c_len = 256
max_t_len = 32
batch_size = 16
epochs = 50
steps_per_epoch = None
pad_token_id = -100

# bert配置
config_path = '/data/pretrain_ckpt/google/mt5-base/bert4torch_config.json'
checkpoint_path = '/data/pretrain_ckpt/google/mt5-base/pytorch_model.bin'
# 下面两个config是从bert4keras中拿的，项目连接https://github.com/bojone/t5_in_bert4keras
spm_path = '/data/pretrain_ckpt/__tensorflow_weights/mt5_bert4keras/sentencepiece_cn.model'
keep_tokens_path = '/data/pretrain_ckpt/__tensorflow_weights/mt5_bert4keras/sentencepiece_cn_keep_tokens.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))


model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    pad_token_id=pad_token_id  # 也可以指定custom_attention_mask并传入attention_mask来实现
).to(device)


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        res = model.decoder.predict([output_ids] + inputs)
        return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

    def generate(self, text, top_k=1):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.beam_search(encoder_output, top_k=top_k)[0]  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])

autotitle = AutoTitle(bos_token_id=0, eos_token_id=tokenizer._token_end_id, max_new_tokens=max_t_len, device=device)


if __name__ == '__main__':
    s1 = u'抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。'
    s2 = u'普适计算环境中未知移动节点的位置信息是定位服务要解决的关键技术。在普适计算二维空间定位过程中,通过对三角形定位单元区域的误差分析,提出了定位单元布局(LUD)定理。在此基础上,对多个定位单元布局进行了研究,定义了一个新的描述定位单元中定位参考点覆盖效能的物理量——覆盖基,提出了在误差最小情况下定位单元布局的覆盖基定理。仿真实验表明定位单元布局定理能更好地满足对普适终端实时定位的需求,且具有较高的精度和最大覆盖效能。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))
