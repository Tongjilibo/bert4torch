from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding
import numpy as np

dict_path = 'E:/data/pretrain_ckpt/google-bert/bert-base-chinese/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def preprocess(text_list):
    batch_token_ids, batch_segment_ids = [], []
    for text in text_list:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=512)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids, length=512)
    batch_segment_ids = sequence_padding(batch_segment_ids, length=512)
    return batch_token_ids, batch_segment_ids

def postprocess(res):
    '''后处理
    '''
    mapping = {0: 'negtive', 1: 'positive'}
    result = []
    for item in res['outputs']:
        prob = np.array(item['data']).reshape(item['shape'])
        pred = prob.argmax(axis=-1)
        result.append([mapping[i] for i in pred])
    return result