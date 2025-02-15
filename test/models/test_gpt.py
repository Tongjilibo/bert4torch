'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(root_path):
    config_path = root_path + '/bert4torch_config.json'
    checkpoint_path = root_path + '/pytorch_model.bin'
    dict_path = root_path + '/bert4torch_vocab.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = build_transformer_model(config_path, checkpoint_path).to(device)

    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    encoder.eval()
    return encoder.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/thu-coai/CDial-GPT_LCCC-base',
                                       'E:/data/pretrain_ckpt/thu-coai/CDial-GPT_LCCC-large'])
@torch.inference_mode()
def test_gpt(model_dir):
    encoder, tokenizer = get_bert4torch_model(model_dir)
    speakers = [tokenizer.token_to_id('[speaker1]'), tokenizer.token_to_id('[speaker2]')]

    class ChatBot(AutoRegressiveDecoder):
        """基于随机采样的闲聊回复
        """
        @AutoRegressiveDecoder.wraps(default_rtype='logits')
        def predict(self, inputs, output_ids, states):
            token_ids, segment_ids = inputs
            curr_segment_ids = torch.zeros_like(output_ids) + token_ids[0, -1]
            token_ids = torch.cat([token_ids, output_ids], 1)
            segment_ids = torch.cat([segment_ids, curr_segment_ids], 1)
            logits = encoder.predict([token_ids, segment_ids])
            return logits[:, -1, :]

        def response(self, texts, n=1, top_k=1):  # topk设置为1表示确定性回复
            token_ids = [tokenizer._token_start_id, speakers[0]]
            segment_ids = [tokenizer._token_start_id, speakers[0]]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
                token_ids.extend(ids)
                segment_ids.extend([speakers[i % 2]] * len(ids))
                segment_ids[-1] = speakers[(i + 1) % 2]
            results = self.random_sample([token_ids, segment_ids], n=n, top_k=top_k)  # 基于随机采样
            return tokenizer.decode(results[0].cpu().numpy())


    chatbot  = ChatBot(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=32, device=device)

    res = chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样'])
    print(res)
    assert res in {'你不是一个人', '你会失去我的'}


if __name__=='__main__':
    test_gpt('E:/data/pretrain_ckpt/thu-coai/CDial-GPT_LCCC-base')