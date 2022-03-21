#! -*- coding: utf-8 -*-
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：https://kexue.fm/archives/7867
# 细节请看：https://github.com/bojone/t5_in_bert4keras
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import glob
torch.manual_seed(42)

# 基本参数
max_c_len = 256
max_t_len = 32
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

def collate_fn(batch):
    """单条样本格式：content：[CLS]文章[SEP]  tgt: [CLS]标题[SEP]
    """
    batch_content_ids, batch_titile_ids = [], []
    for txt in batch:
        text = open(txt, encoding='utf-8').read()
        text = text.split('\n')
        if len(text) > 1:
            content = '\n'.join(text[1:])
            token_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
            batch_content_ids.append(token_ids)  # src用[CLS]开头，[SEP]结尾

            title = text[0]
            token_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
            batch_titile_ids.append(token_ids)  # tgt用[CLS]开头，[SEP]结尾

    # batch_content_ids = torch.tensor([[   2, 1660, 2245, 4659, 4536, 1723,  652, 4282,    1,  575, 1647, 7136,
    #     3187, 5194, 4536, 1290, 4803,  705, 4313, 3073, 3326,  602,   15, 1231,
    #     1891, 4939, 1297,  680, 1367,  705, 4313, 4939, 1297,  680, 1270, 5441,
    #     6720, 6019,  569, 5175,  705, 4313,  409,  750, 1660, 3198,  661, 2556,
    #     1003,  576,   15, 4939, 1297, 3224, 1062, 7342, 6104, 6750, 5175, 1301,
    #     576,  705, 4313,   15, 5343, 2245, 4659, 4536, 3073, 3326,  577, 5441,
    #     3905, 6537, 6719, 4803, 7342, 3622,  409, 2888, 1037,  647,  569, 4803,
    #     1723,  652, 4282,    1,  575, 1647, 7136, 3187, 5194, 4536, 1812, 5175,
    #     705, 4313, 1812, 6926, 4581, 4939, 1297, 3073, 3326,  409, 2088, 3073,
    #     3326, 4536, 3531, 4700, 2493, 6720, 6019,  647, 6293, 3107,  408, 2088,
    #     3073, 3326, 4536, 2026,  957, 2493, 6720, 6019,  647, 1044, 3256,  409,
    #     3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #     0,    0,    0,    0,    0,    0,    0,    0],
    # [   2, 2888, 1037,  647,  569, 4803, 4398,  652,  643, 5491,   64, 5194,
    #     1643,  906, 1044, 5000, 4536, 5008, 5031, 4766, 5205, 3153, 5441, 3073,
    #     3689,   15, 6319, 3073, 3689, 3119,  569, 4803, 3819, 1292, 3153, 5441,
    #     6267, 4948, 2723, 3216,  409, 7572,  942,  784, 4398, 3461, 5026, 1643,
    #     906, 1803, 4313, 4948, 3689, 3239, 2888, 7668, 3044,  600, 1231, 1891,
    #     1643,  906, 4536, 2088, 3581, 2326,  707, 2888, 1255, 2595,  967, 6535,
    #     1175, 1716,  707, 1248, 1770, 2385, 1175, 1716, 6702, 5255,   30, 4095,
    #     1298, 2354, 4887, 4027, 2326,  964, 4393, 4660, 7245,   15, 2888, 1255,
    #     1037, 6032, 2417, 2595,  967, 6535, 1175, 1716, 5190, 4313, 4536, 4192,
    #     2417, 2145, 2493,   30, 2868, 4606,  784, 4398, 5008, 5031, 7313, 3073,
    #     3689, 6720, 6019, 2145, 2493, 5174, 4940, 2298,  670, 4393, 6124, 1054,
    #     30, 3195, 1298,   15, 6290, 6267, 1037, 5008, 5031, 4766, 5205, 5279,
    #     5215,   15, 4398, 3239, 2097, 2595,  967, 6535, 1175, 1716, 1175, 1044,
    #     609, 5577, 2493, 2670, 3119, 2524, 2493,  409,  609,  647, 2088, 2690,
    #     2888, 1037, 4536, 5008, 5031, 7313, 4766, 5205, 5279, 5215, 6720, 6019,
    #     2493, 5441, 6295,  715,   15, 2088, 5633, 2295,  643, 5491,   64, 5194,
    #     1643,  906, 3314, 3213, 6720, 6019,  647, 3742, 6305,   15, 2039, 7639,
    #     5208, 3260, 6032, 3107,   29, 4398, 6319, 3073, 3689, 6720, 6019,  643,
    #     4515, 6297, 1064, 4536, 3044,  758, 1012, 4700, 4270, 6104, 7668,  652,
    #     784, 4398,  969,  698, 2723, 3216,  409,    3]]).to(device)

    # batch_titile_ids = torch.tensor([[   2, 4282,   66,   40,   54,  575, 1647, 7136, 3187, 5194, 4536, 1812,
    #     5175,  705, 4313, 1812, 6926, 4581, 4939, 1297],
    # [   2, 5008, 5031, 4766, 5205, 3153, 5441, 4440,  747,  643, 4515, 1643,
    #     906, 1044, 5000, 3073, 3689, 4675, 4853,    3]]).to(device)
    # return [[batch_content_ids], [batch_titile_ids]], batch_titile_ids[:, 1:].flatten()

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids), dtype=torch.long, device=device)
    return [[batch_content_ids], [batch_titile_ids[:, :-1]]], batch_titile_ids[:, 1:].flatten()

train_dataloader = DataLoader(ListDataset(glob.glob('F:/Projects/data/corpus/文本分类/THUCNews/*/*.txt')), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='t5.1.0',
    segment_vocab_size=0,
    attention_scale=False,
    is_dropout=True,
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
).to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, y_true):
        _, _, y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1.5e-5))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0] 
        return model.predict([[token_ids], [output_ids]])[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, maxlen=max_t_len, device=device)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.pt')
        # 演示效果
        just_show()
    

def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))

if __name__ == '__main__':

    evaluator = Evaluator()
    just_show()
    # model.fit(
    #     train_dataloader,
    #     steps_per_epoch=10,
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

else:
    model.load_weights('./best_model.pt')
