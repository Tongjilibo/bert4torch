#! -*- coding: utf-8 -*-
# 工具函数

import unicodedata
import six
import numpy as np
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import math
import gc
import inspect
import json
import torch.nn.functional as F
import random
from torch4keras.snippets import *


is_py2 = six.PY2

if not is_py2:
    basestring = str

def take_along_dim(input_tensor, indices, dim=None):
    '''兼容部分低版本pytorch没有torch.take_along_dim
    '''
    if torch.__version__ >= '1.9.0':
        return torch.take_along_dim(input_tensor, indices, dim)
    else:
        # 该逻辑仅在少量数据上测试，如有bug，欢迎反馈
        if dim is None:
            res = input_tensor.flatten()[indices]
        else:
            res = np.take_along_axis(input_tensor.cpu().numpy(), indices.cpu().numpy(), axis=dim)
            res = torch.from_numpy(res).to(input_tensor.device)
        # assert res.equal(torch.take_along_dim(input_tensor, indices, dim))
        return res


def torch_div(input, other, round_mode=None):
    # torch.div兼容老版本
    if torch.__version__ <= '1.7.1':
        indices = input // other  # 兼容老版本
    else:
        indices = torch.div(input, other, rounding_mode=round_mode)  # 行索引
    return indices


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)
    

def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text, maxlen, seps='\n', strips=None, truncate=True):
    """将文本按照标点符号划分为若干个短句
       truncate: True表示标点符号切分后仍然超长时, 按照maxlen硬截断分成若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
        return texts
    elif truncate and (not seps) and (len(text) > maxlen):
        # 标点符号用完，仍然超长，且设置了truncate=True
        return [text[i*maxlen:(i+1)*maxlen] for i in range(0, int(np.ceil(len(text)/maxlen)))]
    else:
        return [text]


def merge_segmentate(sequences, maxlen, sep=''):
    '''把m个句子合并成不超过maxlen的n个句子, 主要用途是合并碎句子
    '''
    sequences_new = []
    text = ''
    for t in sequences:
        if text and len(text + sep + t) <= maxlen:
            text = text + sep + t
        elif text:
            sequences_new.append(text)
            text = t
        elif len(t) < maxlen: # text为空
            text = t
        else:
            sequences_new.append(t)
            text = ''
    if text:
        sequences_new.append(text)
    return sequences_new


def text_augmentation(texts, noise_dict=None, noise_len=0, noise_p=0.0, skip_words=None, strategy='random', allow_dup=True):
    '''简单的EDA策略, 增删改
    texts: 需要增强的文本/文本list
    noise_dict: 噪音数据, 元素为str的list, tuple, set
    noise_len: 噪音长度, 优先试用
    noise_p: 噪音比例
    skip_words: 跳过的短语, string/list
    strategy: 修改的策略, 包含增insert, 删delete, 改replace, 随机random
    allow_dup: 是否允许同一个位置多次EDA
    '''
    def insert(text, insert_idx, noise_dict):
        text = list(text)
        for i in insert_idx:
            text[i] = text[i] + random.choice(noise_dict)
        return ''.join(text)

    def delete(text, delete_idx):
        text = list(text)
        for i in delete_idx:
            text[i] = ''
        return ''.join(text)

    def replace(text, replace_idx, noise_dict):
        text = list(text)
        for i in replace_idx:
            text[i] = random.choice(noise_dict)
        return ''.join(text)

    def search(pattern, sequence, keep_last=True):
        """从sequence中寻找子串pattern, 返回符合pattern的id集合
        """
        n = len(pattern)
        pattern_idx_set = set()
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                pattern_idx_set = pattern_idx_set.union(set(range(i, i+n))) if keep_last else pattern_idx_set.union(set(range(i, i+n-1)))
        return pattern_idx_set

    if (noise_len==0) and (noise_p==0):
        return texts

    assert strategy in {'insert', 'delete', 'replace', 'random'}, 'EDA strategy only support insert, delete, replace, random'

    if isinstance(texts, str):
        texts = [texts]

    if skip_words is None:
        skip_words = []
    elif isinstance(skip_words, str):
        skip_words = [skip_words]

    for id, text in enumerate(texts):
        sel_len = noise_len if noise_len > 0 else int(len(text)*noise_p) # 噪声长度
        skip_idx = set()  # 不能修改的idx区间
        for item in skip_words:
            # insert时最后一位允许插入
            skip_idx = skip_idx.union(search(item, text, strategy!='insert'))

        sel_idxs = [i for i in range(len(text)) if i not in skip_idx]  # 可供选择的idx区间
        sel_len = sel_len if allow_dup else min(sel_len, len(sel_idxs))  # 无重复抽样需要抽样数小于总样本
        if (sel_len == 0) or (len(sel_idxs) == 0):  # 如果不可采样则跳过
            continue
        sel_idx = np.random.choice(sel_idxs, sel_len, replace=allow_dup)
        if strategy == 'insert':
            texts[id] = insert(text, sel_idx, noise_dict)
        elif strategy == 'delete':
            texts[id] = delete(text, sel_idx)
        elif strategy == 'replace':
            texts[id] = replace(text, sel_idx, noise_dict)
        elif strategy == 'random':
            if random.random() < 0.333:
                skip_idx = set()  # 不能修改的idx区间
                for item in skip_words:
                    # insert时最后一位允许插入
                    skip_idx = skip_idx.union(search(item, text, keep_last=False))
                texts[id] = insert(text, sel_idx, noise_dict)
            elif random.random() < 0.667:
                texts[id] = delete(text, sel_idx)
            else:
                texts[id] = replace(text, sel_idx, noise_dict)
    return texts if len(texts) > 1 else texts[0]


def lowercase_and_normalize(text, never_split=()):
    """转小写，并进行简单的标准化
    """
    if is_py2:
        text = unicode(text)
    
    # convert non-special tokens to lowercase
    escaped_special_toks = [re.escape(s_tok) for s_tok in never_split]
    pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
    text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    # text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
    
    elif isinstance(inputs[0], torch.Tensor):
        assert mode == 'post', '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数（主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数（主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen, minlen=1, device='cpu'):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.device = device
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含: 1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(self, inputs, output_ids, states, temperature=1, rtype=default_rtype):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (nn.Softmax(dim=-1)(prediction[0] / temperature), prediction[1])
                elif temperature != 1:
                    probas = torch.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return torch.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数
        说明: 定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，
             其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
             rtype=logits时则返回softmax前的结果或者概率对数。
        返回: 二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def beam_search(self, inputs_raw, topk, states=None, temperature=1, min_ends=1, add_btz_dim=True):
        """beam search解码
        说明: 这里的topk即beam size；
        返回: 最优解码序列。
        """
        inputs = []
        for i in inputs_raw:
            if isinstance(i, torch.torch.Tensor):
                pass
            elif isinstance(i, (list, tuple, np.ndarray)) and add_btz_dim:
                i = torch.tensor([i], device=self.device)
            elif isinstance(i, (list, tuple, np.ndarray)) and not add_btz_dim:
                i = torch.tensor(i, device=self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(i)

        output_ids, output_scores = self.first_output_ids, torch.zeros(1, device=self.device)
        for step in range(self.maxlen):
            scores, states = self.predict(inputs, output_ids, states, temperature, 'logits')  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [i.repeat([topk]+[1]*(len(i.shape)-1)) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.flatten().argsort(dim=-1, descending=True)[:topk]  # 仅保留topk
            indices_1 = torch_div(indices, scores.shape[1], rounding_mode='floor')  # 兼容老版本
            # if torch.__version__ <= '1.7.1':
            #     indices_1 = indices // scores.shape[1]  # 兼容老版本
            # else:
            #     indices_1 = torch.div(indices, scores.shape[1], rounding_mode='floor')  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
            output_scores = take_along_dim(scores, indices, dim=None)  # 更新得分
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]

    def random_sample(self, inputs, n, topk=None, topp=None, states=None, temperature=1, min_ends=1):
        """随机采样n个结果
        说明: 非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回: n个解码序列组成的list。
        """
        inputs = [torch.tensor([i], device=self.device) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(inputs, output_ids, states, temperature, 'probas')  # 计算当前概率
            probas /= probas.sum(dim=-1, keepdims=True)  # 确保归一化
            if step == 0:  # 第1步预测后将结果重复n次
                probas = probas.repeat([n]+[1]*(len(probas.shape)-1))
                inputs = [i.repeat([n]+[1]*(len(i.shape)-1)) for i in inputs]
                output_ids = output_ids.repeat([n]+[1]*(len(output_ids.shape)-1))
            if topk is not None:
                k_indices = probas.argsort(dim=-1, descending=True)[:, :topk]  # 仅保留topk
                probas = take_along_dim(probas, k_indices, dim=1)  # topk概率
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(dim=-1, descending=True)  # 从高到低排序
                probas = take_along_dim(probas, p_indices, dim=-1)  # 排序概率
                cumsum_probas = torch.cumsum(probas, dim=-1)  # 累积概率
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的torch.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化

            sample_func = lambda p: torch.multinomial(p, 1)  # 按概率采样函数
            sample_ids = torch.stack([sample_func(p) for p in probas])
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = take_along_dim(p_indices, sample_ids, dim=1)  # 对齐原id
            if topk is not None:
                sample_ids = take_along_dim(k_indices, sample_ids, dim=1)  # 对齐原id
            output_ids = torch.cat([output_ids, sample_ids], 1)  # 更新输出
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        return results


def search_layer(model, layer_name, retrun_first=True):
    '''根据layer_name搜索并返回参数/参数list
    '''
    return_list = []
    for name, param in model.named_parameters():
        if param.requires_grad and layer_name in name:
            return_list.append(param)
    if len(return_list) == 0:
        return None
    if retrun_first:
        return return_list[0]
    else:
        return return_list


class ListDataset(Dataset):
    '''数据是List格式Dataset，支持传入file_path或者外部已读入的data(List格式)
    '''
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path


class IterDataset(IterableDataset):
    '''流式读取文件，用于大数据量、多小文件
       使用时候需要注意steps_per_epoch != None
    '''
    def __init__(self, file_path=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.file_path = file_path
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')
    
    def __iter__(self):
        return self.load_data(self.file_path)

    @staticmethod
    def load_data(file_path, verbose=0):
        if isinstance(file_path, (tuple, list)):
            for file in file_path:
                if verbose != 0:
                    print("Load data: ", file)
                with open(file, 'r') as file_obj:
                    for line in file_obj:
                        yield line
        elif isinstance(file_path, str):
            with open(file_path, 'r') as file_obj:
                for line in file_obj:
                    yield line


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' sinusoid编码
        Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

    # 第二种实现
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids


def cal_ts_num(tensor_shape):
    '''查看某个tensor在gc中的数量
    '''
    cal_num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj): # or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj
            else:
                continue
            if tensor.is_cuda and tensor.size() == tensor_shape:
                print(tensor.shape)
                cal_num+=1
        except Exception as e:
            print('A trivial exception occured: {}'.format(e))
    print(cal_num)


def get_kw(cls, kwargs):
    '''保留排除cls的入参后的kwargs
    '''
    kwargs_new = {}
    for k in kwargs:
        if k not in set(inspect.getargspec(cls)[0]):
            kwargs_new[k] = kwargs[k]
    return kwargs_new


class FGM():
    '''对抗训练
    '''
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    '''对抗训练
    '''
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False, **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            # 修复如pooling层参与foward，但是不参与backward过程时grad为空的问题
            if param.requires_grad and (param.grad is not None):
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None):
                param.grad = self.grad_backup[name]


class VAT():
    '''虚拟对抗训练 https://github.com/namisan/mt-dnn/blob/v0.2/alum/adv_masked_lm.py
    '''
    def __init__(self, model, emb_name='word_embeddings', noise_var=1e-5, noise_gamma=1e-6, adv_step_size=1e-3, 
                 adv_alpha=1, norm_type='l2', **kwargs):
        self.model = model
        self.noise_var = noise_var  # 噪声的方差
        self.noise_gamma = noise_gamma # eps
        self.adv_step_size = adv_step_size  # 学习率
        self.adv_alpha = adv_alpha  # 对抗loss的权重
        self.norm_type = norm_type  # 归一化方式
        self.embed = None
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out
        return None
    
    def forward_(self, train_X, new_embed):
        # 把原来的train_X中的token_ids换成embedding形式
        if isinstance(train_X, (tuple, list)):
            new_train_X = [new_embed] + train_X[1:]
            adv_output = self.model.forward(*new_train_X) if self.model.forward.__code__.co_argcount >= 3 else self.model.forward(new_train_X)
        elif isinstance(train_X, torch.Tensor):
            adv_output = self.model.forward(new_embed)
        return adv_output

    def virtual_adversarial_training(self, train_X, logits):
        # 初始扰动 r
        noise = self.embed.data.new(self.embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        # x + r
        new_embed = self.embed.data.detach() + noise
        adv_output = self.forward_(train_X, new_embed)  # forward第一次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        # 梯度消失，退出
        if torch.isnan(norm) or torch.isinf(norm):
            return None
        # inner sum
        noise = noise + delta_grad * self.adv_step_size
        # projection
        noise = self.adv_project(noise, norm_type=self.norm_type, eps=self.noise_gamma)
        new_embed = self.embed.data.detach() + noise
        new_embed = new_embed.detach()
        # 在进行一次训练
        adv_output = self.forward_(train_X, new_embed)  # forward第二次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        # 在预训练时设置为10，下游任务设置为1
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        return adv_loss
    
    @staticmethod
    def kl(inputs, targets, reduction="sum"):
        """
        计算kl散度
        inputs：tensor，logits
        targets：tensor，logits
        """
        loss = F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction=reduction)
        return loss

    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        """
        L0,L1,L2正则，对于扰动计算
        """
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction


class WebServing(object):
    """简单的Web接口
    用法：
        arguments = {'text': (None, True), 'n': (int, False)}
        web = WebServing(port=8864)
        web.route('/gen_synonyms', gen_synonyms, arguments)
        web.start()
        # 然后访问 http://127.0.0.1:8864/gen_synonyms?text=你好
    说明：
        基于bottlepy简单封装，仅作为临时测试使用，不保证性能。
        目前仅保证支持 Tensorflow 1.x + Keras <= 2.3.1。
        欢迎有经验的开发者帮忙改进。
    依赖：
        pip install bottle
        pip install paste
        （如果不用 server='paste' 的话，可以不装paste库）
    """
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        """封装为接口函数
        参数：
            func：要转换为接口的函数，需要保证输出可以json化，即需要
                  保证 json.dumps(func(inputs)) 能被执行成功；
            arguments：声明func所需参数，其中key为参数名，value[0]为
                       对应的转换函数（接口获取到的参数值都是字符串
                       型），value[1]为该参数是否必须；
            method：GET或者POST。
        """
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        """添加接口
        """
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        """启动服务
        """
        self.bottle.run(host=self.host, port=self.port, server=self.server)


def get_pool_emb(hidden_state=None, pooler=None, attention_mask=None, pool_strategy='cls', custom_layer=None):
    ''' 获取句向量
    '''
    if pool_strategy == 'pooler':
        return pooler
    elif pool_strategy == 'cls':
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} strategy request tensor hidden_state'
        return hidden_state[:, 0]
    elif pool_strategy in {'last-avg', 'mean'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / attention_mask
    elif pool_strategy in {'last-max', 'max'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = hidden_state * attention_mask[:, :, None]
        return torch.max(hid, dim=1)
    elif pool_strategy == 'first-last-avg':
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        hid = torch.sum(hidden_state[1] * attention_mask[:, :, None], dim=1) # 这里不取0
        hid += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)
    elif pool_strategy == 'custom':
        # 取指定层
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        assert isinstance(custom_layer, (int, list, tuple)), f'{pool_strategy} pooling strategy request int/list/tuple custom_layer'
        custom_layer = [custom_layer] if isinstance(custom_layer, int) else custom_layer
        hid = 0
        for i, layer in enumerate(custom_layer, start=1):
            hid += torch.sum(hidden_state[layer] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (i * attention_mask)
    else:
        raise ValueError('pool_strategy illegal')


def parallel_apply_generator(func, iterable, workers, max_queue_size, dummy=False, random_seeds=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。结果将作为一个
    generator返回，其中每个item是输入的序号以及该输入对应的处理结果。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子。
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        """单步函数包装成循环执行
        """
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                while out_queue.qsize() > max_queue_size:
                    yield out_queue.get()
                    out_count += 1
        if out_queue.qsize() > 0:
            yield out_queue.get()
            out_count += 1

    while out_count != in_count:
        yield out_queue.get()
        out_count += 1

    pool.terminate()


def parallel_apply(func, iterable, workers, max_queue_size, callback=None, dummy=False, random_seeds=True, unordered=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。
    参数：
        callback: 处理单个输出的回调函数；
        dummy: False是多进程/线性，True则是多线程/线性；windows需设置dummy=True
        random_seeds: 每个进程的随机种子；
        unordered: 若为False，则按照输入顺序返回，仅当callback为None时生效。
    """
    generator = parallel_apply_generator(func, iterable, workers, max_queue_size, dummy, random_seeds)

    if callback is None:
        if unordered:
            return [d for i, d in generator]
        else:
            results = sorted(generator, key=lambda d: d[0])
            return [d for i, d in results]
    else:
        for i, d in generator:
            callback(d)