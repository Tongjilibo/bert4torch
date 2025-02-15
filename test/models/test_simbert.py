'''simbert'''
import pytest
import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, get_pool_emb
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.tokenizers import Tokenizer, load_vocab
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(model_dir):
    config_path = model_dir + "/bert4torch_config.json"
    if not os.path.exists(config_path):
        config_path = model_dir + "/config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'
    dict_path = model_dir + '/vocab.txt'
    
    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    # 建立加载模型
    class Model(BaseModel):
        def __init__(self, pool_method='cls'):
            super().__init__()
            self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool='linear', application='unilm', keep_tokens=keep_tokens)
            self.pool_method = pool_method

        def forward(self, token_ids, segment_ids):
            hidden_state, pooler, seq_logit = self.bert([token_ids, segment_ids])
            sen_emb = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
            return seq_logit, sen_emb

    model = Model(pool_method='pooler').to(device)

    class SynonymsGenerator(AutoRegressiveDecoder):
        """seq2seq解码器
        """
        @AutoRegressiveDecoder.wraps('logits')
        def predict(self, inputs, output_ids, states):
            token_ids, segment_ids = inputs
            token_ids = torch.cat([token_ids, output_ids], 1)
            segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
            seq_logit, _ = model.predict([token_ids, segment_ids])
            return seq_logit[:, -1, :]

        def generate(self, text, n=1, top_k=5):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=64)
            output_ids = self.random_sample([token_ids, segment_ids], n=n, top_k=top_k)  # 基于随机采样
            return [tokenizer.decode(ids.cpu().numpy()) for ids in output_ids]


    synonyms_generator = SynonymsGenerator(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=64, device=device)

    model.eval()
    return model.to(device), tokenizer, synonyms_generator



def cal_sen_emb(model, tokenizer, text_list):
    '''输入text的list，计算sentence的embedding
    '''
    X, S = [], []
    for t in text_list:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = torch.tensor(sequence_padding(X), dtype=torch.long, device=device)
    S = torch.tensor(sequence_padding(S), dtype=torch.long, device=device)
    _, Z = model.predict([X, S])
    return Z
    

@pytest.mark.parametrize("model_dir", ["E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_tiny",
                                       "E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_small",
                                       "E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_base",
                                       "E:/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_base"])
@torch.inference_mode()
def test_simbert(model_dir):
    query = '我想去首都北京玩玩'
    model, tokenizer, generator = get_bert4torch_model(model_dir)

    gen_texts = generator.generate(query, top_k=1)[0]
    print(gen_texts)
    assert gen_texts in {'北京北京去北京玩，我想去北京，怎么办理',
                         '我想去北京玩玩，有什么好玩的地方推荐吗？',
                         '我想去北京玩，有什么好玩的地方吗？',
                         }


    target_text = '我想去首都北京玩玩'
    text_list = ['我想去北京玩', '北京有啥好玩的吗？我想去看看', '好渴望去北京游玩啊']
    Z = cal_sen_emb(model, tokenizer, [target_text]+text_list)
    assert Z.sum().item() - 6.9725 < 1e-4


if __name__=='__main__':
    test_simbert("E:/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_base")