#! -*- coding: utf-8 -*-
# 预训练语料构建，这里实现的mlm任务的，NSP和SOP未使用
# 方案：一直动态生成文件，超过最大保存数目时候sleep，
# 当训练速度超过文件生成速度时候，可开启多个数据生成脚本

import numpy as np
from bert4torch.tokenizers import Tokenizer
import json, glob, re
from tqdm import tqdm
import collections
import gc
import shelve
import time
import os
import random
import jieba
jieba.initialize()


class TrainingDataset(object):
    """预训练数据集生成器
    """
    def __init__(self, tokenizer, sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0
        """
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text):
        """单个文本的处理函数，返回处理后的instance
        """
        raise NotImplementedError

    def paragraph_process(self, texts, starts, ends, paddings):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；starts是每个instance的起始id；
              ends是每个instance的终止id；paddings是每个instance的填充id。
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。
        """
        instances, instance = [], [[start] for start in starts]

        for text in texts:
            # 处理单个句子
            sub_instance = self.sentence_process(text)
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]
            new_length = len(instance[0]) + len(sub_instance[0])

            # 如果长度即将溢出
            if new_length > self.sequence_length - 1:
                # 插入终止符，并padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)
                    item = self.padding(item, pad)
                    complete_instance.append(item)
                # 存储结果，并构建新样本
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # 样本续接
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)

        return instances

    def serialize(self, instances, db, count):
        """写入到文件
        """
        for instance in instances:
            input_ids, masked_lm_labels = instance[0], instance[1]
            assert len(input_ids) <= sequence_length
            features = collections.OrderedDict()
            features["input_ids"] = input_ids
            features["masked_lm_labels"] = masked_lm_labels
            db[str(count)] = features
            count += 1
        return count

    def process(self, corpus, record_name):
        """处理输入语料（corpus）
        """
        count = 0

        db = shelve.open(record_name)
        for texts in corpus:
            instances = self.paragraph_process(texts)
            count = self.serialize(instances, db, count)
            
        db.close()
        del instances
        gc.collect()

        # 记录对应的文件名和样本量
        record_info = {"filename": record_name, "samples_num": count}
        json.dump(record_info, open(record_name + ".json", "w", encoding="utf-8"))

        print('write %s examples into %s' % (count, record_name))


class TrainingDatasetRoBERTa(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）
    """
    def __init__(self, tokenizer, word_segment, mask_rate=0.15, sequence_length=512):
        """参数说明：
            tokenizer必须是bert4torch自带的tokenizer类；
            word_segment是任意分词函数。
        """
        super(TrainingDatasetRoBERTa, self).__init__(tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列, 来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            

            if rand < self.mask_rate:
                word_mask_ids = [self.token_process(i) for i in word_token_ids]
                token_ids.extend(word_mask_ids)
                mask_ids.extend(word_token_ids)
            else:
                token_ids.extend(word_token_ids)
                word_mask_ids = [0] * len(word_tokens)
                mask_ids.extend(word_mask_ids)
                
        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBERTa, self).paragraph_process(texts, starts, ends, paddings)


if __name__ == '__main__':
    sequence_length = 512  # 文本长度
    max_file_num = 40  # 最大保存的文件个数
    dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'  # 字典文件
    dir_training_data = 'E:/Github/bert4torch/examples/datasets/pretrain'  # 保存的文件目录
    dir_corpus = 'F:/Projects/data/corpus/pretrain'  # 读入的语料地址
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        '''挑选语料
        '''
        files_corpus = glob.glob(f'{dir_corpus}/*/*')  # 根据目录结构自行调整
        file_corpus = random.choice(files_corpus)  # 随机挑选一篇文章
        count, texts = 0, []

        with open(file_corpus, encoding='utf-8') as f:
            for l in tqdm(f, desc=f'Load data from {file_corpus}'):
                l = l.strip()
                texts.extend(re.findall(u'.*?[\n。]+', l))
                count += 1
                if count == 10:  # 10篇文章合在一起再处理
                    yield texts
                    count, texts = 0, []
        if texts:
            yield texts

    def word_segment(text):
        return jieba.lcut(text)

    TD = TrainingDatasetRoBERTa(tokenizer, word_segment, sequence_length=sequence_length)

    while True:
        train_files = [file for file in os.listdir(dir_training_data) if ('train_' in file) and ('dat' in file)]
        # 当保存的训练文件未达到指定数量时
        if len(train_files) < max_file_num:
            record_name = f'{dir_training_data}/train_'+ time.strftime('%Y%m%d%H%M%S', time.localtime())
            TD.process(corpus=some_texts(), record_name=record_name)
            time.sleep(1)  # 可不加，这里是防止生成文件名一样
        else:
            time.sleep(300)

