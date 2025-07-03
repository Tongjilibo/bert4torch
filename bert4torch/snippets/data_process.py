#! -*- coding: utf-8 -*-
'''工具函数处理模块
'''

import unicodedata
import six
import numpy as np
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List, Tuple, Callable, Iterable, Set, Literal
from torch4keras.snippets import log_warn
import random
import os
from tqdm import tqdm

is_py2 = six.PY2

if not is_py2:
    basestring = str


def is_string(s):
    """判断是否是字符串"""
    return isinstance(s, basestring)
    

def truncate_sequences(sequences:Iterable[List[int]], maxlen:int, indices:Union[int, List[int], Tuple[int]]=-1):
    """截断各个sequences以保证总长度至不超过maxlen, 原地修改，优先从最长的sequence开始截断
    :param sequences: List[List[int]], 需要截断的序列
    :param maxlen: int, 所有序列的总长度
    :param indices: int/List[int]/Tuple[int] 每次去掉的token_id的index

    ### Example
    ```python
    from bert4torch.snippets import truncate_sequences
    seq = [list(range(20)), list(range(30))]
    res = truncate_sequences(seq, maxlen=11, indices=-1)
    print(res, seq)
    # 输出：[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]] [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
    ```
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)
    assert len(indices) == len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text:str, maxlen:int, seps:str='\n', strips:str=None, truncate:bool=True, greed:bool=False):
    """将文本按照标点符号划分为若干个短句
       
    :param text: 待划分的句子
    :param maxlen: int, 截断长度
    :param seps: 分隔符
    :param strips: ''.strip()
    :param truncate: True表示标点符号切分后仍然超长时, 按照maxlen硬截断分成若干个短句
    :param greed: bool, 是否贪婪切分
        - True表示seps之间没有优先级，从左向右寻找以任意sep结尾且尽量填满maxlen的片段，如果找不到则以maxlen硬截断，切分后每个片段比较满
        - False表示按照seps从前往后的优先级切分，容易切的比较碎
    :return: List[str], 划分后的句子列表

    ### Example
    ```python
    from bert4torch.snippets import text_segmentate
    from pprint import pprint
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    text = '''
    你好，大家好！我叫小明，是一名刚刚踏出大学校门，满怀憧憬与热情的新晋毕业生。在过去的几年里，我在XX大学深造，专业聚焦于XX领域，这段学习经历不仅为我打下了坚实的理论基础，也让我在实践中积累了宝贵的经验。
    在校期间，我积极参与各类学术科研活动，曾参与XX项目的研究，这段经历锻炼了我的问题解决能力和团队合作精神。同时，我还担任了学生会的XX职位，负责组织策划了多场校园活动，这些经历极大地提升了我的组织协调能力和领导力，也让我学会了如何在压力下保持高效工作。
    除了专业学习和社会实践，我还热衷于XX技能/爱好，比如编程、摄影或是公共演讲，这不仅丰富了我的大学生活，也让我在兴趣中找到了自我成长的另一种可能。
    现在，我带着对未知世界的好奇和对职业发展的渴望，站在了人生的新起点上。我期望能够将所学应用到实际工作中，为团队带来创新思维和活力，同时也期待在新的工作环境中不断学习，实现个人价值与公司目标的双赢。
    最后，非常感谢有这个机会向大家介绍自己，我期待着与大家一起成长，共同面对挑战，创造美好的未来。谢谢大家！
    '''
    maxlen = 50
    res = text_segmentate(text, maxlen, seps, strips, greed=False)
    pprint(res)

    # 输出
    # ['你好，大家好！我叫小明，是一名刚刚踏出大学校门，满怀憧憬与热情的新晋毕业生。',
    # '在过去的几年里，我在XX大学深造，专业聚焦于XX领域，这段学习经历不仅为我打下了坚实的理论基础',
    # '也让我在实践中积累了宝贵的经验。',
    # '在校期间，我积极参与各类学术科研活动，曾参与XX项目的研究',
    # '这段经历锻炼了我的问题解决能力和团队合作精神。',
    # '同时，我还担任了学生会的XX职位，负责组织策划了多场校园活动',
    # '这些经历极大地提升了我的组织协调能力和领导力，也让我学会了如何在压力下保持高效工作。',
    # '除了专业学习和社会实践，我还热衷于XX技能/爱好，比如编程、摄影或是公共演讲',
    # '这不仅丰富了我的大学生活，也让我在兴趣中找到了自我成长的另一种可能。',
    # '现在，我带着对未知世界的好奇和对职业发展的渴望，站在了人生的新起点上。',
    # '我期望能够将所学应用到实际工作中，为团队带来创新思维和活力，同时也期待在新的工作环境中不断学习',
    # '实现个人价值与公司目标的双赢。',
    # '最后，非常感谢有这个机会向大家介绍自己，我期待着与大家一起成长，共同面对挑战，创造美好的未来。',
    # '谢谢大家！']

    res = text_segmentate(text, maxlen, seps, strips, greed=True)
    pprint(res)

    # 输出
    # ['你好，大家好！我叫小明，是一名刚刚踏出大学校门，满怀憧憬与热情的新晋毕业生。在过去的几年里',
    # '我在XX大学深造，专业聚焦于XX领域，这段学习经历不仅为我打下了坚实的理论基础',
    # '也让我在实践中积累了宝贵的经验。\n在校期间，我积极参与各类学术科研活动，曾参与XX项目的研究',
    # '这段经历锻炼了我的问题解决能力和团队合作精神。同时，我还担任了学生会的XX职位',
    # '负责组织策划了多场校园活动，这些经历极大地提升了我的组织协调能力和领导力',
    # '也让我学会了如何在压力下保持高效工作。\n除了专业学习和社会实践，我还热衷于XX技能/爱好',
    # '比如编程、摄影或是公共演讲，这不仅丰富了我的大学生活，也让我在兴趣中找到了自我成长的另一种可能。\n',
    # '现在，我带着对未知世界的好奇和对职业发展的渴望，站在了人生的新起点上。',
    # '我期望能够将所学应用到实际工作中，为团队带来创新思维和活力，同时也期待在新的工作环境中不断学习',
    # '实现个人价值与公司目标的双赢。\n最后，非常感谢有这个机会向大家介绍自己，我期待着与大家一起成长',
    # '共同面对挑战，创造美好的未来。谢谢大家！']
    ```
    """
    text = text.strip().strip(strips)
    if not greed:
        if seps and len(text) > maxlen:
            pieces = text.split(seps[0])  # 按照最优先级的sep截断
            text, texts = '', []
            for i, p in enumerate(pieces):
                if text and p and len(text) + len(p) > maxlen - 1:  # text+当前piece后超长了
                    texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
                    text = ''
                if i + 1 == len(pieces):  # 最后一个片段
                    text = text + p
                else:
                    text = text + p + seps[0]  # 追加到当前text
            if text:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
            return texts
        elif truncate and (not seps) and (len(text) > maxlen):
            # 标点符号用完，仍然超长，且设置了truncate=True
            return [text[i*maxlen:(i+1)*maxlen] for i in range(0, int(np.ceil(len(text)/maxlen)))]
        else:
            return [text]
    else:
        texts = ['']
        chunk = ''
        for char in text:
            if char in seps:
                if len(texts[-1]) + len(chunk) < maxlen:
                    texts[-1] += chunk + char
                    chunk = ''
                else:
                    texts.append(chunk + char)
                    chunk = ''
            else:
                if len(chunk) < maxlen:
                    chunk += char
                else:
                    texts.append(chunk[:maxlen])
                    chunk = chunk[maxlen:] + char
        texts = [text.strip(strips) for text in texts]
        return texts


def merge_segmentate(sequences:List[str], maxlen:int, sep:str=''):
    '''把m个句子合并成不超过maxlen的n个句子, 主要用途是合并碎句子

    :param sequences: List(str), 短句子列表
    :param maxlen: int, 最大长度
    :param sep: str, 合并使用的分隔符, 可以是，。等标点符号
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


def text_augmentation(texts:Union[str, List[str]], noise_dict:Union[List[str], Tuple[str], Set[str]]=None, noise_len:int=0, noise_p:float=0.0, 
                      skip_words:Union[str, List[str]]=None, strategy:Literal['random', 'insert', 'delete', 'replace']='random', allow_dup:bool=True):
    '''简单的EDA策略, 增删改
    
    :param texts: 需要增强的文本/文本list
    :param noise_dict: 噪音数据, 元素为str的list, tuple, set
    :param noise_len: 噪音长度, 优先试用
    :param noise_p: 噪音比例
    :param skip_words: 跳过的短语, string/list
    :param strategy: 修改的策略, 包含增insert, 删delete, 改replace, 随机random
    :param allow_dup: 是否允许同一个位置多次EDA
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
        """从sequence中寻找子串pattern, 返回符合pattern的id集合"""
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


def lowercase_and_normalize(text:str, never_split:Union[Set, Tuple, List]=()):
    """转小写，并进行简单的标准化"""
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


def sequence_padding(inputs:Union[List[np.ndarray], List[List], List[torch.Tensor]], length:Union[List, int]=None, 
                     value:int=0, seq_dims:int=1, padding_side:Literal['left', 'right']='right', mode=None):
    """将序列padding到同一长度"""
    if mode is not None:
        raise DeprecationWarning('Args `mode` has been deprecated since v0.5.5, use `padding_side` instead')
    
    def flatten(lst):
        """Flatten a nested list."""
        for item in lst:
            item:torch.Tensor
            if item.dim() > 1:
                yield from flatten(item)
            else:
                yield item
    
    return_torch_format = False
    if all([isinstance(input_, torch.Tensor) for input_ in inputs]) :
        # 都是torch.Tensor        
        inputs = list(flatten(inputs))  # 多维度的则打平处理
        if length is not None:
            inputs = [i[:length] for i in inputs]
        
        if padding_side == 'right':
            return pad_sequence(inputs, padding_value=value, batch_first=True)
        else:
            # 转为np.array处理
            inputs = [i.numpy() for i in inputs]
            return_torch_format = True

    if all([isinstance(input_, (np.ndarray, list)) for input_ in inputs]):
        # 都是np.ndarray, list
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
                if padding_side == 'right':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif padding_side == 'left':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "right" or "left".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return torch.tensor(np.array(outputs)) if return_torch_format else np.array(outputs)
    
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def parallel_apply_generator(func:Callable, iterable:Iterable, workers:int, max_queue_size:int, dummy:bool=False, random_seeds:bool=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。结果将作为一个
    generator返回，其中每个item是输入的序号以及该输入对应的处理结果。
    
    :param func: 对iterable进行处理的回调函数；
    :param iterable: 供func处理的输入项，为一个可迭代对象；
    :param workers: 并行处理的workers数量；
    :param max_queue_size: quene的最大数量；
    :param dummy: False是多进程/线性，True则是多线程/线性；
    :param random_seeds: 每个进程的随机种子。
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
        """单步函数包装成循环执行"""
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


def parallel_apply(func:Callable, iterable:Iterable, workers:int, max_queue_size:int, callback:Callable=None, 
                   dummy:bool=False, random_seeds:bool=True, unordered:bool=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。

    :param func: 对iterable进行处理的回调函数；
    :param iterable: 供func处理的输入项，为一个可迭代对象；
    :param workers: 并行处理的workers数量；
    :param max_queue_size: quene的最大数量；
    :param callback: 处理单个输出的回调函数；
    :param dummy: False是多进程/线性，True则是多线程/线性；windows需设置dummy=True
    :param random_seeds: 每个进程的随机种子；
    :param unordered: 若为False，则按照输入顺序返回，仅当callback为None时生效。
    """
    if (os.name == 'nt') and (dummy is False):
        log_warn('Args `dummay` shold be True when in window os')

    generator = parallel_apply_generator(func, iterable, workers, max_queue_size, dummy, random_seeds)

    results = []
    for i, d in tqdm(generator, desc='ParallelApply'):
        results.append((i, d))
    
    if unordered:
        results = [d for i, d in results]
    else:
        results = sorted(results, key=lambda d: d[0])
        results = [d for i, d in results]
    
    if callback is not None:
        results = [callback(d) for d in results]
    return results


def parallel_apply_concurrent(func:Callable, iterable:Iterable, workers:int, max_queue_size:int, 
                              callback:Callable=None, dummy:bool=False, unordered:bool=True, **kwargs):
    '''使用concurrent来并发执行
    :param func: 对iterable进行处理的回调函数；
    :param iterable: 供func处理的输入项，为一个可迭代对象；
    :param workers: 并行处理的workers数量；
    :param batchsize: batchsize大小；
    :param callback: 处理单个输出的回调函数；
    :param dummy: False是多进程/线性，True则是多线程/线性；windows需设置dummy=True
    :param random_seeds: 每个进程的随机种子；
    :param unordered: 若为False，则按照输入顺序返回，仅当callback为None时生效。
    '''
    import concurrent.futures
    results = []

    if dummy:
        Executor = concurrent.futures.ThreadPoolExecutor
    else:
        Executor = concurrent.futures.ProcessPoolExecutor

    # 如果没有长度则变成列表
    if not hasattr(iterable, '__len__'):
        iterable = [i for i in iterable]

    with Executor(max_workers=workers) as executor:
        futures = []
        for i in range(0, len(iterable), max_queue_size):
            batch = iterable[i:i + max_queue_size]
            future = executor.submit(func, batch)
            futures.append((i, future))
        
        for i, future in tqdm(futures, desc='ParallelApplyConcurrent'):
            result = future.result()
            results += [(i+j, res) for j, res in enumerate(result)]
    
    if unordered:
        results = [d for i, d in results]
    else:
        results = sorted(results, key=lambda d: d[0])
        results = [d for i, d in results]


    if callback is not None:
        return [callback(d) for d in results]
    
    return results

PoolStrategy = Literal['pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'last-token', 'custom']
def get_pool_emb(hidden_state:Union[list, tuple, torch.Tensor]=None, pooled_output:torch.Tensor=None, attention_mask:torch.Tensor=None, 
                 pool_strategy:PoolStrategy='cls', custom_layer:Union[int, List[int]]=None):
    ''' 获取句向量

    :param hidden_state: torch.Tensor/List(torch.Tensor)，last_hidden_state/all_encoded_layers
    :param pooled_output: torch.Tensor, bert的pool_output输出
    :param attention_mask: torch.Tensor
    :param pool_strategy: str, ('pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom')
        pooler: 使用bert的pooler输出
        cls: 使用[CLS]的输出
        last-avg/mean: 最后一层的输出做average pooling
        last-max/max: 最后一层的输出做max pooling
        first-last-avg: 第一层和最后一层输出做average pooling
        last-token: 使用最后一个token的输出
        custom: 自定义的层数做average pooling
    :param custom_layer: int/List[int], 指定对某几层做average pooling
    '''
    if pool_strategy == 'pooler':
        # 使用bert的pooler输出
        if pooled_output is None:
            log_warn('Args `pooled_output` is None')
        return pooled_output
    
    elif pool_strategy == 'cls':
        # 使用[CLS]的输出
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pool_strategy request tensor hidden_state'
        return hidden_state[:, 0]
    
    elif pool_strategy in {'last-avg', 'mean'}:
        # 最后一层的输出做average pooling
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pool_strategy request tensor hidden_state'
        hid = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / attention_mask
    
    elif pool_strategy in {'last-max', 'max'}:
        # 最后一层的输出做max pooling
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pool_strategy request tensor hidden_state'
        hid = torch.masked_fill(hidden_state, (1-attention_mask[:, :, None]).bool(), torch.finfo(hidden_state.dtype).min)
        return torch.max(hid, dim=1).values
    
    elif pool_strategy == 'first-last-avg':
        # 第一层和最后一层输出做average pooling
        assert isinstance(hidden_state, list), f'{pool_strategy} pool_strategy request list hidden_state'
        hid = torch.sum(hidden_state[1] * attention_mask[:, :, None], dim=1) # 这里不取0
        hid += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)

    elif pool_strategy == 'last-token':
        # 使用最后一个token的输出
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        bs, seq_len, hidden_dim = hidden_state.shape
        values, indices = attention_mask.flip(1).max(1)
        indices = torch.where(values == 0, seq_len - 1, indices)
        gather_indices = seq_len - indices - 1

        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
        gather_indices = gather_indices.unsqueeze(1)
        assert gather_indices.shape == (bs, 1, hidden_dim)

        # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
        # Actually no need for the attention mask as we gather the last token where attn_mask = 1
        # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
        # use the attention mask to ignore them again
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_state.size()).to(hidden_state.dtype)
        )
        embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        return embedding
    
    elif pool_strategy == 'custom':
        # 取指定层
        assert isinstance(hidden_state, list), f'{pool_strategy} pool_strategy request list hidden_state'
        assert isinstance(custom_layer, (int, list, tuple)), f'{pool_strategy} pool_strategy request int/list/tuple custom_layer'
        custom_layer = [custom_layer] if isinstance(custom_layer, int) else custom_layer
        hid = 0
        for i, layer in enumerate(custom_layer, start=1):
            hid += torch.sum(hidden_state[layer] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (i * attention_mask)
    else:
        raise ValueError(f'Args `pool_strategy`={pool_strategy} not supported')


def create_position_ids_start_at_padding(input_ids, padding_idx, past_key_values_length=0, start_padding_idx=True):
    """生成padding_ids, 从padding_idx+1开始。忽略填充符号"""
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask    
    return incremental_indices.long() + (padding_idx if start_padding_idx else 0)
