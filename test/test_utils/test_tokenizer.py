'''测试tokenizer和transformers自带的tokenizer是否一致，测试后是一致的
'''
import pytest
from transformers import BertTokenizer, XLNetTokenizer, XLNetTokenizerFast
from bert4torch.tokenizers import Tokenizer, SpTokenizer
from tqdm import tqdm
import os


def compare(data_path, tokenizer, tokenizer_b4t, truncation=False):
    with open(data_path, 'r', encoding='utf-8') as f:
        for l in tqdm(f):
            l = l.split('\t')[0].strip()
            tokens1 = tokenizer.tokenize(l)
            tokens2 = tokenizer_b4t.tokenize(l)
            tokens2 = tokens2[1:-1] if truncation else tokens2
            assert tokens1 == tokens2, ''.join(tokens1) + ' <------> ' + ''.join(tokens2)


@pytest.mark.parametrize("model_dir", ["E:/data/pretrain_ckpt/google-bert/bert-base-chinese"])
@pytest.mark.parametrize("data_path", ['F:/data/corpus/sentence_classification/sentiment/sentiment.train.data'])
def test_bert_tokenizer(model_dir, data_path):
    '''测试bert的tokenizer'''
    print('Test BertTokenizer')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    tokenizer_b4t = Tokenizer(os.path.join(model_dir, 'vocab.txt'), do_lower_case=True, do_tokenize_unk=True)
    compare(data_path, tokenizer, tokenizer_b4t, truncation=True)


@pytest.mark.parametrize("model_dir", ["E:/data/pretrain_ckpt/hfl/chinese-xlnet-base"])
@pytest.mark.parametrize("data_path", ['F:/data/corpus/sentence_classification/sentiment/sentiment.train.data'])
def test_xlnet_tokenizer(model_dir, data_path):
    '''测试xlnet的tokenizer'''
    print('Test SpTokenizer')
    tokenizer = XLNetTokenizer.from_pretrained(model_dir)
    # tokenizer = XLNetTokenizerFast.from_pretrained(model_dir)  # fast版本有些许不一样
    tokenizer_b4t = SpTokenizer(os.path.join(model_dir, 'spiece.model'), token_start=None, token_end=None)
    compare(data_path, tokenizer, tokenizer_b4t)


if __name__ == '__main__':
    test_bert_tokenizer("E:/data/pretrain_ckpt/google-bert/bert-base-chinese", 'F:/data/corpus/sentence_classification/sentiment/sentiment.train.data')
    test_xlnet_tokenizer("E:/data/pretrain_ckpt/hfl/chinese-xlnet-base", 'F:/data/corpus/sentence_classification/sentiment/sentiment.train.data')