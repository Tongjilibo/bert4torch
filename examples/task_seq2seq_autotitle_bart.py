#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用encoder-decoder方案
# 训练时候收敛较慢，比unilm方案慢不少

from bert4pytorch.models import build_transformer_model, BaseModel
from bert4pytorch.tokenizers import Tokenizer, load_vocab
from bert4pytorch.snippets import sequence_padding, text_segmentate
from bert4pytorch.snippets import AutoRegressiveDecoder, Callback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert4pytorch.snippets import ListDataset
import glob

torch.manual_seed(1234)

# 基本参数
max_c_len = 512
max_t_len = 32
batch_size = 12
epochs = 10000

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bart/[FudanNLP_torch_base]/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
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

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids), dtype=torch.long, device=device)
    return [[batch_content_ids], [batch_titile_ids[:, :-1]]], batch_titile_ids[:, 1:].flatten()

train_dataloader = DataLoader(ListDataset(glob.glob('F:/Projects/data/corpus/文本分类/THUCNews/*/*.txt')), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只保留keep_tokens中的字，精简原字表
        self.seq2seq_model = build_transformer_model(config_path, checkpoint_path, model='bart', keep_tokens=keep_tokens, segment_vocab_size=0)
        self.tgt_word_prj = nn.Linear(768, len(token_dict), bias=True)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if kwargs.get('tgt_emb_prj_weight_sharing'):
            # decoder底层的embedding和顶层的全连接共享
            self.tgt_word_prj.weight = self.tgt_embeddings.word_embeddings.weight
            self.x_logit_scale = (768 ** -0.5)
        else:
            self.x_logit_scale = 1.
    def forward(self, inputs):
        decoder_hidden_state = self.seq2seq_model(inputs)
        y_pred = self.tgt_word_prj(decoder_hidden_state)
        return y_pred

model = Model().to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1.5e-5))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        return model.predict([[token_ids], [output_ids]])[:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, maxlen=max_t_len, device=device)

def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))

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

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=500,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')
