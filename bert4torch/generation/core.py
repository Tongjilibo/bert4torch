'''自回归模型的生成
'''

from typing import Union, Optional, List, Tuple, Literal
import torch
import torch.nn as nn
import numpy as np
from bert4torch.generation.logits_process import *
from bert4torch.snippets import take_along_dim, torch_div, sequence_padding, create_position_ids_start_at_padding, \
    log_info, log_warn, log_warn_once, inference_mode
from bert4torch.tokenizers import TokenizerBase
from torch4keras.model import BaseModel
from torch4keras.trainer import Trainer
from contextlib import contextmanager
import gc


INPUT_TYPE = Union[torch.Tensor, List[Union[int, list, tuple, torch.Tensor, np.ndarray]], Tuple[Union[int, list, tuple, torch.Tensor, np.ndarray]], np.ndarray]
TUPLE_LIST_TENSOR_TYPE = Union[Tuple[torch.Tensor], List[torch.Tensor]]


def get_max_input_seqlen(input_seqlen:torch.Tensor):
    # 获取一个tensor中最大的值
    return input_seqlen.max().item() if len(input_seqlen) > 1 else input_seqlen.item()


class EmptyCacheDecorators(object):
    optimize_cuda_cache = False

    @classmethod
    @contextmanager
    def empty_cuda_cache(cls):
        yield
        if cls.optimize_cuda_cache and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略

    :param bos_token_id: int, 解码使用的起始token_id, 不同预训练模型设置可能不一样
    :param eos_token_id: int/tuple/list, 解码使用的结束token_id, 不同预训练模型设置可能不一样, 默认给的-1(真实场景中不存在, 表示输出到max_length)
    :param max_new_tokens: int, 最大解码长度
    :param min_new_tokens: int, 最小解码长度, 默认为1
    :param max_length: int, 最大文本长度
    :param pad_token_id: int, pad_token_id, 在batch解码时候使用
    :param padding_side: str, padding在前面还是后面, left或者right
    :param device: str, 默认为'cpu'
    :param n: int, random_sample时候表示生成的个数; beam_search时表示束宽
    :param top_k: int, 这里的top_k是指仅保留topk的值 (仅在top_k上进行概率采样)
    :param top_p: float, 这里的top_p是token的概率阈值设置(仅在头部top_p上进行概率采样)
    :param temperature: float, 温度参数, 默认为1, 越小结果越确定, 越大结果越多样
    :param repetition_penalty: float, 重复的惩罚系数, 越大结果越不重复
    :param min_ends: int, 最小的end_id的个数
    :param return_last_token: bool, 在stream_generate模式下, 是否仅输出last_token, 默认为False表示输出解码出来的历史token
        1) 理论上stream模式下, 应该只返回last_token, 但由于有的模型的tokenizer单个字符会被拆分, 只输出last_token会显示乱码
        2) 可以设置为True的情形: 一是tokenize对于字符不会拆分的情况(乱码); 二是tokenizer=None时, 返回的是last_token_id, 用户自行decode也可以
    :param return_states: bool, 是否返回缓存的states, 主要是有history模式下, 返回past_key_values有利于加速

    """
    @inference_mode()
    def __init__(self, 
                 bos_token_id:int=None, 
                 eos_token_id:int=-1, 
                 max_new_tokens:int=None, 
                 min_new_tokens:int=1, 
                 max_length:int=64, 
                 pad_token_id:int=0, 
                 padding_side:Literal['left', 'right']='right', 
                 device:str='cpu', 
                 n:int=1, 
                 top_k:int=None, 
                 top_p:float=None,
                 temperature:float=1.0, 
                 repetition_penalty:float=1.0, 
                 do_sample:bool=True,
                 min_ends:int=1, 
                 **generation_config):
        # generation_config
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.max_length = max_length  # 最大长度, 含输入的text
        self.pad_token_id = pad_token_id   # pad_token_id兼容bert4torch和hf的, 如错误则需要显式传入pad_id:int
        self.padding_side = padding_side
        self.device = device

        # 生成的样本个数
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"`n` has to be a strictly positive integer, but is {n}")
        self.n = n  # 为每条样本生成几个输出
        self.top_k = top_k  # topk采样
        self.top_p = top_p  # top_p采样
        self.temperature = temperature  # 温度系数
        self.repetition_penalty = repetition_penalty  # 重复性惩罚系数
        self.do_sample = do_sample  # 是否采样, 如果为False则不进行top_k和top_p采样, 直接取最大值
        self.prepared_logits_processor = self._get_logits_processor()
        self.min_ends = min_ends
        self.return_last_token = False
        self.return_states = False
        
        for old_key, new_key in {'start_id': 'bos_token_id', 'end_id': 'eos_token_id', 'topk': 'top_k', 
                                 'topp': 'top_p', 'maxlen': 'max_length', 'pad_id': 'pad_token_id'}.items():
            if old_key in generation_config:
                raise DeprecationWarning(f'Args `{old_key}` not supported since 0.2.8, use {new_key} instead')
        self.set_generation_config(generation_config)
                
        self.use_batch = False
        if self.bos_token_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.bos_token_id]], device=device)

    def set_generation_config(self, kwargs):
        if kwargs.get('generation_config'):
            generation_config = kwargs.pop('generation_config')
            kwargs.update(**generation_config)
    
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)  # 在generate()时候动态设置属性
            # else:
            #     log_warn_once(f'Generation_config `{key}` has not been pre maintained')
        self._set_logits_processor()

    def _get_logits_processor(self):
        processors = LogitsProcessorList()
        if not self.do_sample or self.temperature == 0:  # greed
            return processors
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=self.repetition_penalty))
        processors.append(TemperatureLogitsWarper(self.temperature))
        processors.append(TopKLogitsWarper(top_k=self.top_k))
        processors.append(TopPLogitsWarper(top_p=self.top_p))
        return processors
    
    def _set_logits_processor(self):
        for processor in self.prepared_logits_processor:
            if isinstance(processor, RepetitionPenaltyLogitsProcessor):
                processor.penalty = self.repetition_penalty
            elif isinstance(processor, TemperatureLogitsWarper):
                processor.temperature = self.temperature
            elif isinstance(processor, TopKLogitsWarper):
                processor.top_k = self.top_k
            elif isinstance(processor, TopPLogitsWarper):
                processor.top_p = self.top_p

    @staticmethod
    def wraps(default_rtype:Literal['probas', 'logits']='probas', use_states:bool=False):
        """用来进一步完善predict函数
        :param default_rtype: probas不会进行softmax处理，logits会进行softmax处理
        :param use_states: bool, 表示是否使用cache

        目前包含: 
            1. 设置rtype参数, 并做相应处理; 
            2. 确定states的使用, 并做相应处理; 
            3. 设置温度参数, 并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, states:Optional[dict], rtype=default_rtype):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)  # logits / (logits, states)

                if use_states:
                    assert len(prediction) == 2, 'Should return 2 output when set use_states=True'
                    logits, states = prediction
                else:
                    logits, states = prediction, None

                # prepared_logits_processor
                # past_token_ids: 包含query部分的input_ids
                input_ids_or_encoder_hidden_states = inputs[0]
                logits = self.prepared_logits_processor(input_ids_or_encoder_hidden_states, logits, output_ids=output_ids, states=states)    

                if default_rtype == 'logits':
                    logits = nn.functional.softmax(logits, dim=-1)  # 转为概率

                if rtype == 'probas':
                    return logits, states
                else:
                    return torch.log(logits + 1e-12), states

            # 增加函数, 用于动态修改闭包中的属性
            def set_default_rtype(value):
                nonlocal default_rtype
                default_rtype = value
            new_predict.set_default_rtype = set_default_rtype

            def set_use_states(value):
                nonlocal use_states
                use_states = value
            new_predict.set_use_states = set_use_states

            return new_predict

        return actual_decorator

    def predict(self, inputs, output_ids:torch.Tensor, states:dict=None):
        """用户需自定义递归预测函数; 
        说明: 定义的时候, 需要用wraps方法进行装饰, 传入default_rtype和use_states, 其中default_rtype为字符串logits或probas, probas时返回归一化的概率, 
        rtype=logits时则返回softmax前的结果或者概率对数。
        
        :return: 二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def _trans2tensors(self, inputs_raw: INPUT_TYPE) -> list:
        '''对当前输入进行处理, 并都转化成tensor, return: list[tensor]
        :param inputs_raw: tensor/list(tensor)/list(list)/list(int)
        '''
        if isinstance(inputs_raw, torch.Tensor):
            # 传入的Tensor直接[]后返回
            self.input_seqlen = torch.ones(inputs_raw.shape[0], dtype=torch.long).to(self.device) * inputs_raw.shape[1]
            # input_seqlen = get_max_input_seqlen(self.input_seqlen)  # 超长警告
            # if input_seqlen >= self.max_length:
            #     log_warn(f'prompt_len={input_seqlen} >= max_length={self.max_length}')
            return [inputs_raw.to(self.device)]
        elif isinstance(inputs_raw, (tuple,list)) and all([isinstance(i, int) for i in inputs_raw]):
            # List[int]
            inputs_raw = [inputs_raw]
        
        inputs = []
        for input_ in inputs_raw:
            if isinstance(input_, torch.Tensor):
                # encoder-decoder传入的encoder_hidden_states和encoder_attention_mask
                input_new = input_
                self.input_seqlen = torch.zeros(input_new.shape[0], dtype=torch.long).to(self.device)
            elif isinstance(input_, (list, tuple, np.ndarray)):
                # 单条样本为[1,2,3]格式, 需转为[[1,2,3]]
                input_ = input_ if self.use_batch else [input_]
                input_new = torch.tensor(sequence_padding(input_, value=self.pad_token_id, padding_side=self.padding_side), dtype=torch.long, device=self.device)

                # padding在右边则input_seqlen是真实的长度, 左边则统一为最大程度
                if self.padding_side == 'right':
                    self.input_seqlen = torch.tensor([len(i) for i in input_], dtype=torch.long).to(self.device)
                else:
                    max_len = input_new.shape[1]
                    self.input_seqlen = torch.tensor([max_len]*len(input_new), dtype=torch.long).to(self.device)
            else:
                raise TypeError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(input_new)
        return inputs
    
    def _define_stopping_criteria(self, states: Optional[dict]=None):
        ''' max_new_tokens的校准'''
        if (self.max_new_tokens is None) and (self.max_length is None):
            raise ValueError('Args `max_new_tokens` and `max_length` can not both be None')
        elif self.max_new_tokens is not None:
            max_new_tokens = self.max_new_tokens
        elif self.max_new_tokens is None:
            # 这里用max是因为batch_generate时候self.input_seqlen是多个
            input_seqlen = get_max_input_seqlen(self.input_seqlen)
            max_new_tokens = max(0, self.max_length-input_seqlen)
            if input_seqlen >= self.max_length:
                # 超长警告
                log_warn(f'Prompt len={input_seqlen} >= max_length={self.max_length}')

        if (states is not None) and (states.get('past_key_values') is not None):
            past_key_values_lenghth = states.get('past_key_values')[0][0].shape[2]
            max_new_tokens = max(0, max_new_tokens-past_key_values_lenghth)
        
        return range(max_new_tokens)

    def _identify_sentence_end(self, output_ids:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''判断句子是否遇到停止符结束'''
        if isinstance(self.eos_token_id, (int, float)):
            # self.eos_token_id: 2
            is_end = output_ids[:, -1] == self.eos_token_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.eos_token_id).sum(1)  # 统计出现的end标记
        elif isinstance(self.eos_token_id, (set, tuple, list)):
            # self.eos_token_id: [2, 102, 68712]
            end_counts = 0
            is_end = None
            for eos_token_id in self.eos_token_id:
                if isinstance(eos_token_id, (int, float)):  # self.eos_token_id: [2, 102, 68712]
                    tmp = output_ids[:, -1] == eos_token_id
                    is_end = tmp if is_end is None else is_end + tmp
                    end_counts += (output_ids == eos_token_id).sum(1)
                elif isinstance(eos_token_id, (set, tuple, list)):  # self.eos_token_id: [[2, 102], 68712]
                    tmp = torch.tensor([i[-len(eos_token_id):].tolist() == eos_token_id for i in output_ids], device=output_ids.device)
                    is_end = tmp if is_end is None else is_end + tmp
                    end_counts += torch.ones_like(is_end)  # TODO: 目前只有Qwen有这种操作
                else:
                    raise TypeError(f'TypeError: eos_token_id={eos_token_id}')
        return is_end, end_counts

    def __beam_search_step(self, step:int, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, output_scores:torch.Tensor, states: Optional[dict]=None):
        '''beam_search单条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, 'logits')  # 计算当前得分
        if step == 0:  # 第1步预测后将输入重复topk次
            inputs = [i.repeat([self.top_k]+[1]*(len(i.shape)-1)) for i in inputs]
        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        indices = scores.flatten().argsort(dim=-1, descending=True)[:self.top_k]  # 仅保留topk
        indices_1 = torch_div(indices, scores.shape[1], rounding_mode='floor')  # 兼容老版本
        indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
        output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
        output_scores = take_along_dim(scores, indices, dim=None)  # 更新得分
        return inputs, output_ids, output_scores, states

    def __beam_search_end(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, output_scores:torch.Tensor, results:list):
        '''beam_search单条推理计算是否结束'''
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        flag = ~is_end | (end_counts < self.min_ends)  # 标记未完成序列
        self.flag = flag  # 记录未完成序列
        if output_ids.shape[1] >= self.min_new_tokens:  # 最短长度判断
            best = output_scores.argmax()  # 得分最大的那个
            if is_end[best] and end_counts[best] >= self.min_ends:  # 如果已经终止
                break_tag = True
            elif not flag.all():  # 如果有已完成的
                inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                output_ids = output_ids[flag]  # 扔掉已完成序列
                output_scores = output_scores[flag]  # 扔掉已完成序列
                end_counts = end_counts[flag]  # 扔掉已完成end计数
                self.top_k = flag.sum()  # topk相应变化
        return inputs, output_ids, output_scores, results, break_tag

    def __batch_beam_search_step(self, step:int, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, output_scores:torch.Tensor, states: Optional[dict]=None):
        '''beam_search batch条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, 'logits')  # 计算当前得分
        if step == 0:  # 第0步预测后将输入重复topk次
            inputs_new = []
            for input_ in inputs:
                inputs_ = []
                for top_k, input_i in zip(self.top_k, input_):
                    input_i = input_i.unsqueeze(0)
                    inputs_.append(input_i.repeat([top_k]+[1]*(len(input_i.shape)-1)))
                inputs_new.append(torch.cat(inputs_))
            inputs = inputs_new
            # 对seq_len进行扩充
            input_seqlen = []
            for top_k, input_seqlen_i in zip(self.top_k, self.input_seqlen):
                input_seqlen.append(input_seqlen_i.repeat(top_k))
            self.input_seqlen = torch.cat(input_seqlen)

        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        output_ids_new, output_scores_new = [], []
        for smp_i, top_k in enumerate(self.top_k):
            if step == 0:
                score = scores[smp_i][None, ...]
                output_id = output_ids[smp_i][None, ...]
            else:
                start, end = sum(self.top_k[:smp_i]), sum(self.top_k[:smp_i+1])
                score = scores[start:end]
                output_id = output_ids[start:end]

            indices = score.flatten().argsort(dim=-1, descending=True)[:top_k]  # 仅保留topk
            indices_1 = torch_div(indices, score.shape[1], rounding_mode='floor')  # 兼容老版本
            indices_2 = (indices % score.shape[1]).reshape((-1, 1))  # 列索引
            output_id = torch.cat([output_id[indices_1], indices_2], 1)  # 更新输出
            output_score = take_along_dim(score, indices, dim=None)  # 更新得分
            output_ids_new.append(output_id)
            output_scores_new.append(output_score)
        output_ids_new = torch.cat(output_ids_new)
        output_scores_new = torch.cat(output_scores_new)
        return inputs, output_ids_new, output_scores_new, states

    def __batch_beam_search_end(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, output_scores:torch.Tensor, results:list):
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        self.flag = ~is_end | (end_counts < self.min_ends)  # 标记未完成序列

        if output_ids.shape[1] >= self.min_new_tokens:  # 最短长度判断
            inputs_new, output_ids_new, output_scores_new, flag_new = [], [], [], []
            topks_new = self.top_k.copy()
            for smp_i, top_k in enumerate(self.top_k): # 这里的top_k是一个list
                if top_k == 0:
                    continue
                start, end = sum(self.top_k[:smp_i]), sum(self.top_k[:smp_i+1])
                input_ = [i[start:end] for i in inputs]
                output_score = output_scores[start:end]
                output_id = output_ids[start:end]

                best = output_score.argmax()  # 得分最大的那个
                is_end, end_counts = self._identify_sentence_end(output_id)
                flag = ~is_end | (end_counts < self.min_ends)  # 标记未完成序列
                if is_end[best] and end_counts[best] >= self.min_ends:  # 如果已经终止
                    results[smp_i] = output_id[output_score.argmax()]
                    flag = torch.zeros_like(flag, dtype=torch.bool)
                if not flag.all():  # 如果有已完成的
                    input_ = [i[flag] for i in input_]  # 扔掉已完成序列
                    output_id = output_id[flag]  # 扔掉已完成序列
                    output_score = output_score[flag]  # 扔掉已完成序列
                    end_counts = end_counts[flag]  # 扔掉已完成end计数
                    topks_new[smp_i] = flag.sum().item()  # topk相应变化
                
                inputs_new.append(input_)
                output_ids_new.append(output_id)
                output_scores_new.append(output_score)
                flag_new.append(flag)

            inputs = [torch.cat(i) for i in zip(*inputs_new)]
            output_ids = torch.cat(output_ids_new)
            output_scores = torch.cat(output_scores_new)
            self.top_k = topks_new
            self.flag = torch.cat(flag_new)

            if len(output_ids) == 0:
                break_tag = True
        else:
            # 不满足结束条件, 需要对self.flag进行更新
            self.flag = torch.ones_like(self.flag, dtype=torch.bool)
        
        return inputs, output_ids, output_scores, results, break_tag

    def beam_search(self, inputs: INPUT_TYPE, states: Optional[dict]=None, **generation_config):
        """beam search解码
        
        :param inputs: 编码器的输入, 包含encoder_hidden_states, encoder_attention_mask
        :param top_k: int, 这里的top_k即beam size
        :param states: None/dict, 缓存参数
        :param temperature: 温度参数, 默认为1
        :param min_ends:
        :return: 最优解码序列。
        """
        self.set_generation_config(generation_config)
        assert self.top_k is not None, 'Arg `top_k` means beam_size anc can not be None'
        inputs = self._trans2tensors(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1) if btz > 1 else self.first_output_ids
        output_scores = torch.zeros(btz, device=self.device)
        results = []

        # batch推理
        if self.use_batch:
            self.top_k = [self.top_k] * btz
            results = [None] * btz

        for step in self._define_stopping_criteria(states):
            if (not self.use_batch):  # 单条推理
                inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, states)
                inputs, output_ids, output_scores, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, results)
            else:  # batch推理
                inputs, output_ids, output_scores, states = self.__batch_beam_search_step(step, inputs, output_ids, output_scores, states)
                inputs, output_ids, output_scores, results, break_tag = self.__batch_beam_search_end(inputs, output_ids, output_scores, results)
            if break_tag:
                break
        
        # 如果还有未完成序列, 直接放入结果
        if len(output_ids) > 0:
            if not self.use_batch:
                results.append(output_ids[output_scores.argmax()])
            elif self.use_batch:
                for smp_i, result in enumerate(results):
                    if result is None:
                        start, end = sum(self.top_k[:smp_i]), sum(self.top_k[:smp_i+1])
                        output_score = output_scores[start:end]
                        output_id = output_ids[start:end]
                        results[smp_i] = output_id[output_score.argmax()]

        # 达到长度直接输出
        self.flag = None
        if self.return_states:
            return results, states
        else:
            return results

    def stream_beam_search(self, inputs: INPUT_TYPE, states:Optional[dict]=None, **generation_config):
        '''beam_search的stream输出模式'''
        self.set_generation_config(generation_config)
        assert self.top_k is not None, 'Arg `top_k` means beam_size anc can not be None'
        inputs = self._trans2tensors(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1) if btz > 1 else self.first_output_ids
        output_scores = torch.zeros(btz, device=self.device)
        results = []
        for step in self._define_stopping_criteria(states):
            inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, states)

            if self.return_last_token and self.return_states:
                yield [output_ids[output_scores.argmax()][-1:]], states
            elif self.return_last_token:
                yield [output_ids[output_scores.argmax()][-1:]]  # 仅yield最后一个token
            elif self.return_states:
                yield [output_ids[output_scores.argmax()]], states
            else:
                yield [output_ids[output_scores.argmax()]]
            
            inputs, output_ids, output_scores, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, results)
            if break_tag:
                break 
            
        # 达到长度直接输出
        self.flag = None

    def __random_sample_step(self, step:int, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, states: Optional[dict]=None):
        '''为了random_sample和stream_random_sample共用, 抽离出来的单步逻辑'''
        self.step = step
        probas, states = self.predict(inputs, output_ids, states, 'probas')  # 计算当前概率
        probas: torch.FloatTensor = probas / probas.sum(dim=-1, keepdims=True)  # 确保归一化

        if step == 0 and self.n > 1:  # 第1步预测后将结果重复n次
            probas = probas.repeat([self.n]+[1]*(len(probas.shape)-1))
            inputs = [i.repeat([self.n]+[1]*(len(i.shape)-1)) for i in inputs]
            output_ids = output_ids.repeat([self.n]+[1]*(len(output_ids.shape)-1))
            self.input_seqlen = self.input_seqlen.repeat([self.n, 1])
        
        if not self.do_sample or self.temperature == 0:  # greed输出，和vllm一致，transformers不支持设置，而是采用do_sample=False
            sample_func = lambda p: torch.argmax(p, dim=-1)
        else:
            sample_func = lambda p: torch.multinomial(p, 1)  # 按概率采样函数
        sample_ids = torch.stack([sample_func(p) for p in probas])
        sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状

        output_ids = torch.cat([output_ids, sample_ids], 1)  # 更新输出
        return inputs, output_ids, states

    def __random_sample_end(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, results:list, smp_indexs:torch.Tensor=None):
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        f_flag = is_end & (end_counts >= self.min_ends)  # 标记已完成序列
        self.flag = (f_flag == False)  # 记录未完成序列, 这里是裁前的, 用于use_states时候的裁剪操作
        if (output_ids.shape[1] >= self.min_new_tokens) and f_flag.any():  # 最短长度判断, 如果有已完成的
            if smp_indexs is None:  # single
                for ids in output_ids[f_flag]:  # 存好已完成序列
                    results.append(ids)
            else:  # batch
                for smp_i, ids in zip(smp_indexs[f_flag], output_ids[f_flag]):  # 存好已完成序列
                    results[smp_i] = ids

            flag = (f_flag == False)  # 标记未完成序列, True表示未完成
            inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
            output_ids = output_ids[flag]  # 只保留未完成部分候选集
            # end_counts = end_counts[flag]  # 只保留未完成部分end计数
            if smp_indexs is not None:
                smp_indexs = smp_indexs[flag]
            if len(output_ids) == 0:
                break_tag = True
        else:
            # 不满足结束条件, 需要对self.flag进行更新
            self.flag = torch.ones_like(self.flag, dtype=torch.bool)

        if smp_indexs is None:
            return inputs, output_ids, results, break_tag
        else:
            return inputs, output_ids, results, break_tag, smp_indexs

    def random_sample(self, inputs_raw: INPUT_TYPE, states:Optional[dict]=None, **generation_config):
        """随机采样n个结果; 
        说明: 非None的topk表示每一步只从概率最高的topk个中采样; 而非None的topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        
        :param inputs_raw: tensor、array、list、tuple, 解码的输入, 一般为last_hidden_state, shape=[btz, seq_len, hdsz]
        :param top_k: int, 这里的top_k是指仅保留topk的值
        :param top_p: float, 这里的top_p是token的概率阈值设置
        :param states: None/dict, 缓存参数
        :param temperature: 温度参数, 默认为1
        :param min_ends: 最小的end_id的个数
        :return: n个解码序列组成的list。
        """
        self.set_generation_config(generation_config)
        inputs = self._trans2tensors(inputs_raw)  # 对输入进行处理
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1) if btz > 1 else self.first_output_ids
        results = []

        if self.use_batch:
            smp_indexs = torch.tensor(range(btz)).to(output_ids)
            results = [None] * btz

        for step in self._define_stopping_criteria(states):
            if not self.use_batch:  # single
                inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)
                inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results)
            else:  # batch
                inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)
                inputs, output_ids, results, break_tag, smp_indexs = self.__random_sample_end(inputs, output_ids, results, smp_indexs)
            if break_tag:
                break

        # 如果还有未完成序列, 直接放入结果
        if len(output_ids) > 0:
            if not self.use_batch:
                for ids in output_ids:
                    results.append(ids)
            elif self.use_batch:
                for smp_i, ids in zip(smp_indexs, output_ids):  # 存好已完成序列
                    results[smp_i] = ids
        # 返回结果
        self.flag = None
        if self.return_states:
            return results, states
        else:
            return results
    
    def stream_random_sample(self, inputs_raw: INPUT_TYPE, states:Optional[dict]=None, **generation_config):
        """随机采样n个结果; stream输出"""
        generation_config['n'] = 1
        self.set_generation_config(generation_config)
        inputs = self._trans2tensors(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        results = []
        for step in self._define_stopping_criteria(states):
            inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)

            if self.return_last_token and self.return_states:
                yield output_ids[:, -1:], states
            elif self.return_last_token:
                yield output_ids[:, -1:]  # 仅yield最后一个token
            elif self.return_states:
                yield output_ids, states
            else:
                yield output_ids
            
            inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results)
            if break_tag:
                break
        self.flag = None


class SeqGeneration(AutoRegressiveDecoder):
    '''单向decoder语言模型的解码, 对AutoRegressiveDecoder的简单封装, 可以使用cache来加快解码

    :param model: 模型
    :param tokenizer: tokenizer, 如果使用第三方的tokenize, 需要继承重写下pre_process和post_process
    :param tokenizer_config: dict, tokenize的参数, 一般设置这个即可, 若部分参数无法设置, 则单独设置tokenizer_encode_config和tokenizer_decode_config
    :param tokenizer_encode_config: dict, tokenize的encode参数
    :param tokenizer_decode_config: dict, tokenize的decode参数
    :param default_rtype: str, 模型输出的结果是logits设置为logits, 如果是probas设置为probas
    :param use_states: str, 是否使用cache
    :param device: str, 默认为None, 因为可以直接使用传入的model.device

    > generation_config: 接受两种传参方式, 1)接受两种传参方式, generation_config={'top_k':50}; 2) top_k=50, top_p=0.9, ...
    :param bos_token_id: int, 解码使用的起始token_id, 不同预训练模型设置可能不一样
    :param eos_token_id: int/tuple/list, 解码使用的结束token_id, 不同预训练模型设置可能不一样, 默认给的-1(真实场景中不存在, 表示输出到max_length)
    :param max_new_tokens: int, 最大解码长度
    :param min_new_tokens: int, 最小解码长度, 默认为1
    :param pad_token_id: int, pad_token_id, 在batch解码时候使用
    :param padding_side: str, padding在前面还是后面, left或者right
    :param n: int, random_sample时候表示生成的个数; beam_search时表示束宽
    :param top_k: int, 这里的top_k是指仅保留topk的值
    :param top_p: float, 这里的top_p是token的概率阈值设置
    :param temperature: 温度参数, 默认为1
    :param repetition_penalty: 重复的惩罚系数
    :param min_ends: int, 最小的end_id的个数
    :param include_input: int, 输出是否包含输入
    :param return_last_token: bool, 在stream_generate模式下, 是否仅输出last_token, 默认为False表示输出解码出来的历史token
        1) 理论上stream模式下, 应该只返回last_token, 但由于有的模型的tokenizer单个字符会被拆分, 只输出last_token会显示乱码
        2) 可以设置为True的情形: 一是tokenize对于字符不会拆分的情况(乱码); 二是tokenizer=None时, 返回的是last_token_id, 用户自行decode也可以
    '''
    def __init__(self, 
                 model:BaseModel, 
                 tokenizer=None, 
                 tokenizer_config:dict=None, 
                 mode:Literal['beam_search', 'random_sample']='random_sample', 
                 default_rtype:Literal['logits', 'probas']='logits',
                 use_states:bool=True, 
                 optimize_cuda_cache:bool=False,
                 **kwargs):
        # 如果传进来的是Trainer
        if isinstance(model, Trainer) and hasattr(model, 'module'):
            # log_info(f'Args `model` is a Trainer instance, will `use model.module` instead')
            model = model.module
        
        if model.training:
            model.eval()
        
        if hasattr(model, 'config'):  # 从model.config中解析出generation_config和tokenizer_config
            generation_config = model.config.get('generation_config', dict())
            tokenizer_config = {**generation_config.pop('tokenizer_config', dict()), 
                                **(tokenizer_config if tokenizer_config else dict())}
            kwargs.update({k:v for k, v in generation_config.items() if k not in kwargs})
        kwargs = self._default_generation_config(tokenizer, model, **kwargs)  # 对部分参数进行默认设置
        super().__init__(**kwargs)

        self.encoder = None
        self.decoder = model

        # tokenizer参数
        self.tokenizer = tokenizer
        self.tokenizer_type = 'b4t' if isinstance(tokenizer, TokenizerBase) else 'hf'
        self.tokenizer_encode_config = kwargs.get('tokenizer_encode_config') or self.clear_tokenizer_config(tokenizer_config, 'encode')
        self.tokenizer_decode_config = kwargs.get('tokenizer_decode_config') or self.clear_tokenizer_config(tokenizer_config, 'decode')

        assert mode in {'random_sample', 'beam_search'}, 'Args `mode` only support `random_sample/beam_search`.'
        self.mode = mode
        self.predict.set_default_rtype(default_rtype)  # 动态修改闭包中的default_rtype
        self.predict.set_use_states(use_states)  # 动态修改闭包中的use_states
        self.use_states = use_states
        self.use_segment_ids = hasattr(model, 'segment_vocab_size') and (model.segment_vocab_size > 0)  # 是否使用segment_ids
        self.include_input = False  # 输出是否包含输入
        self.input_text = ''
        EmptyCacheDecorators.optimize_cuda_cache = optimize_cuda_cache
        self.set_generation_config(kwargs)
    
    @staticmethod
    def _default_generation_config(tokenizer, model, **kwargs):
        # 可以直接以generation_config方式传参, 也可以top_p=0.9, top_k=50方式传参
        if kwargs.get('generation_config'):
            generation_config = kwargs.pop('generation_config')
            kwargs.update(**generation_config)

        ''' genration的默认参数设置 '''
        if kwargs.get('pad_token_id') is not None:  # 用户自行设置
            pass
        elif (tokenizer is not None) and (tokenizer.pad_token_id is not None):
            kwargs['pad_token_id'] = tokenizer.pad_token_id
        else:
            kwargs['pad_token_id'] = 0
            log_info(f'Arg `pad_token_id` has been set to default value 0')
        
        if kwargs.get('eos_token_id') is not None:  # 用户自行设置
            pass
        elif (tokenizer is not None) and hasattr(tokenizer, 'eos_token_id') and (tokenizer.eos_token_id is not None):
            kwargs['eos_token_id'] = tokenizer.eos_token_id
            log_info(f'Arg `eos_token_id` has been set to tokenizer.eos_token_id:{tokenizer.eos_token_id}')
        else:
            kwargs['eos_token_id'] = kwargs['pad_token_id']
            log_info(f'Arg `eos_token_id` has been set to `pad_token_id`:{kwargs["pad_token_id"]}')
        kwargs['device'] = kwargs.get('device') or next(model.parameters()).device

        return kwargs

    def _prepare_next_inputs(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, include_past:bool=True, **states):
        '''decode时候准备下一次输入, 使用cache则仅输入last_token_ids, 不适用需要输入past_token_ids'''
        def concat_token_ids(token_ids, output_ids):
            '''把非padding部分concat在一起
            '''
            if output_ids.numel() == 0:
                return token_ids

            if not self.use_batch:
                return torch.cat([token_ids, output_ids], 1)
            
            inputs_ = []
            for seq_l, token_ids_i, output_ids_i in zip(self.input_seqlen, token_ids, output_ids):
                inputs_.append(torch.cat([token_ids_i[:seq_l], output_ids_i, token_ids_i[seq_l:]]))
            return torch.stack(inputs_)

        # 部分序列已经完成, 需要更新input_seqlen
        if len(self.input_seqlen) != len(output_ids):
            self.input_seqlen = self.input_seqlen[self.flag]
        
        if include_past is False:
            # 使用cache且step>=1, 即用last_token做下一步预测
            if len(inputs) == 1:
                next_inputs = [output_ids[:, -1:]]  # last_token
            elif len(inputs) >= 2:  # 第二个是segment_ids
                log_warn_once('Use 0 for follwing segment_ids')
                next_inputs = [output_ids[:, -1:], torch.zeros_like(output_ids[:, -1:])]
        else:
            # 不使用cache, 或使用cache且step==0时
            if len(inputs) == 1:
                token_ids = concat_token_ids(inputs[0], output_ids)
                next_inputs = [token_ids]
            elif len(inputs) >= 2:  # 第二个是segment_ids
                token_ids, segment_ids = inputs
                token_ids = concat_token_ids(token_ids, output_ids)
                log_warn_once('Use 0 for follwing segment_ids')
                segment_ids = concat_token_ids(segment_ids, torch.zeros_like(output_ids))
                next_inputs = [token_ids, segment_ids]
        
        # 方便每个模型自定义
        states.update({'output_ids': output_ids, 'input_seqlen': self.input_seqlen, 'step': self.step})
        states = self.decoder.prepare_inputs_for_generation(*inputs, **states)
        return next_inputs, states

    def __get_last_token_logits(self, logits, output_ids):
        '''获取最后一个token的logits'''
        if not self.use_batch or logits.shape[1] == 1:
            return logits[:, -1, :]
        else:
            # 由于batch是padding过的, 因此要找到padding前的位置
            last_token_index = self.input_seqlen + output_ids.shape[1] - 1
            return torch.stack([logit[index_, :] for index_, logit in zip(last_token_index, logits)])

    @AutoRegressiveDecoder.wraps()
    def predict(self, inputs:TUPLE_LIST_TENSOR_TYPE, output_ids:torch.Tensor, states:Optional[dict]):
        '''
        :param inputs: 原始输入, 在整个预测过程中均不改变
        :param outputs_ids: 输出的ids, 随着预测进行, 逐步增长
        :param states: None/dict, 缓存参数
        '''        
        # 使用cache, 输入只能padding在左侧
        if self.use_states:
            if states is None:
                states = {'use_states': True}
            elif states.get('use_states') is not True:
                states.update({'use_states': True})

            if self.step == 0:
                # next_inputs：step=0时候输入全部
                next_inputs, states = self._prepare_next_inputs(inputs, output_ids, include_past=True, **states)

                # position_ids: 在第0步如果padding在左侧, 则需要自定义position_ids
                if 'position_ids' not in states and self.use_batch and (self.padding_side == 'left'):
                    # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    #         [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])
                    states['position_ids'] = create_position_ids_start_at_padding(next_inputs[0], self.pad_token_id, past_key_values_length=-1, start_padding_idx=False)

            elif self.step >= 1:
                # next_inputs：>=1时候输入last_token
                next_inputs, states = self._prepare_next_inputs(inputs, output_ids, include_past=False, **states)
                
                # position_ids: >=1步则取last_token
                if states.get('position_ids') is not None and states['position_ids'].dim() == 2:
                    states['position_ids'] = states['position_ids'][:, -1:] + 1

                # attention_mask: 根据token_ids生成的, 因此这里重置下
                if states.get('attention_mask_2d') is not None:  # 在states中attention_mask_2d才是[btz, seq_len]
                    attention_mask = states['attention_mask_2d']
                    states['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                    del states['attention_mask_2d']

                # shape和size不一致: step=1时候, beam_search的btz=束宽, 或者要返回多个结果
                if (self.mode == 'beam_search' or self.n > 1) and (self.step == 1):
                    btz = len(self.flag)
                    if (states.get('past_key_values') is not None) and states['past_key_values'][0][0].shape[0] != btz:
                        repeat_size = btz // states['past_key_values'][0][0].shape[0]
                        for l_i, past_key_value in enumerate(states['past_key_values']):
                            states['past_key_values'][l_i] = (past_key_value[0].repeat_interleave(repeat_size, dim=0), past_key_value[1].repeat_interleave(repeat_size, dim=0))
                    if (states.get('cross_past_key_values') is not None) and states['cross_past_key_values'][0][0].shape[0] != btz:
                        repeat_size = btz // states['cross_past_key_values'][0][0].shape[0]
                        for l_i, cross_past_key_value in enumerate(states['cross_past_key_values']):
                            states['cross_past_key_values'][l_i] = (cross_past_key_value[0].repeat_interleave(repeat_size, dim=0), cross_past_key_value[1].repeat_interleave(repeat_size, dim=0))
                    if (states.get('attention_mask') is not None) and states['attention_mask'].shape[0] != btz:
                        repeat_size = btz // states['attention_mask'].shape[0]
                        states['attention_mask'] = states['attention_mask'].repeat_interleave(repeat_size, dim=0)
                    if (states.get('position_ids') is not None) and states['position_ids'].shape[0] != btz:
                        repeat_size = btz // states['position_ids'].shape[0]
                        states['position_ids'] = states['position_ids'].repeat_interleave(repeat_size, dim=0)

                # 已经有序列完成, 仅保留未完成序列(self.flag中True表示未完成序列)
                if hasattr(self, 'flag') and (self.flag is not None) and (self.flag==0).sum().item() > 0:
                    states['attention_mask'] = states['attention_mask'][self.flag]
                    if (len(states['position_ids']) > 1):
                        states['position_ids'] = states['position_ids'][self.flag]
                    states['past_key_values'] = [(kv[0][self.flag], kv[1][self.flag]) for kv in states['past_key_values']]
                    if 'cross_past_key_values' in states:
                        states['cross_past_key_values'] = [(kv[0][self.flag], kv[1][self.flag]) for kv in states['cross_past_key_values']]

            logits, states = self.decoder.predict(next_inputs, **states)
            states = self.decoder._update_model_kwargs_for_generation(logits, states)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return logits[:, -1, :], states

        # 不使用cache
        elif not self.use_states:
            next_inputs, states = self._prepare_next_inputs(inputs, output_ids, include_past=True)

            # 如果use_states=False且padding_side='left', 则需要自定义position_ids, 否则position_ids不正确, 但这么使用很少
            if 'position_ids' not in states and self.padding_side == 'left':
                states['position_ids'] = create_position_ids_start_at_padding(next_inputs[0], self.pad_token_id, past_key_values_length=-1, start_padding_idx=False)
            logits = self.decoder.predict(next_inputs, **states)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return self.__get_last_token_logits(logits, output_ids)

    def clear_tokenizer_config(self, config:dict, mode:str):
        '''获取在tokenize_params中的参数'''
        if (self.tokenizer is None) or (config is None):
            return dict()
        # TODO: 这里allowed_special是qwen独有的, 后续可能要修改
        encode_kwargs = {'max_length', 'add_special_tokens', 'padding', 'truncation', 'stride', 'allowed_special', 'return_tensors', 
                         'maxlen', 'return_dict', 'truncate_from', 'return_offsets'}
        decode_kwargs = {'skip_special_tokens', 'clean_up_tokenization_spaces'}
        if mode == 'encode':
            clear_config = {k:v for k, v in config.items() if k in encode_kwargs}
            if isinstance(clear_config.get('allowed_special', -1), list):
                clear_config['allowed_special'] = set(clear_config['allowed_special'])
            return clear_config
        elif mode == 'decode':
            return {k:v for k, v in config.items() if k in decode_kwargs}
        else:
            raise ValueError(f'Only support `encode` and `decode` options, {mode} not supported')

    def pre_process(self, inputs:Union[str, torch.Tensor, List[Union[int, torch.Tensor]], Tuple[Union[int, torch.Tensor]]]) -> \
                    Union[torch.Tensor, List[Union[torch.Tensor, List[int]]]]:
        '''前处理, 可以继承后自定义, 主要用于第三方tokenizer的encode'''
        self.input_text = inputs if self.include_input else ''
        
        # 以下情形不需要tokenize的，直接返回
        if isinstance(inputs, torch.Tensor):  # tensor
            return inputs
        elif isinstance(inputs, (tuple, list)) and all([isinstance(i, torch.Tensor) for i in inputs]):  # list[tensor]
            return inputs
        elif isinstance(inputs, (tuple, list)) and all([isinstance(i, int) for i in inputs]):  # list[int]
            return [inputs]
        
        # str | List[str] | Tuple[str]类型
        assert isinstance(inputs, str) or (isinstance(inputs, (tuple,list)) and all([isinstance(i, str) for i in inputs]))
        if self.tokenizer is None:
            raise ValueError(f"tokenizer can not be None when query'format is {type(inputs)}")
        
        # 传入的是text或者list(text)
        if self.tokenizer_type == 'b4t':
            # bert4torch的tokenizer
            inputs = self.tokenizer.encode(inputs, **self.tokenizer_encode_config)
            return inputs if self.use_segment_ids else [inputs[0]]
        elif self.tokenizer_type == 'hf':
            # hf的tokenize
            return [self.tokenizer(inputs, **self.tokenizer_encode_config)['input_ids']]
    
    def post_process(self, output_ids:List[torch.Tensor]):
        '''后处理, 可以继承后自定义, 主要用于第三方tokenizer的decode'''
        if self.tokenizer is None:
            return output_ids
        
        if self.return_states:
            output_ids, states = output_ids

        if len(output_ids) > 1:
            output_text = [self.tokenizer.decode(ids.cpu().numpy(), **self.tokenizer_decode_config) for ids in output_ids]
            if isinstance(self.input_text, str):
                # 输入是str, 则在前面直接添加
                output_text = [self.input_text + item for item in output_text]
            elif isinstance(self.input_text, (tuple, list)) and (len(self.input_text) == len(output_text)):
                # 输入是list且和outputs同维度
                output_text = [self.input_text[i] + item for i, item in enumerate(output_text)]
        elif len(output_ids) == 1:
            output_text = self.input_text + self.tokenizer.decode(output_ids[0].cpu().numpy(), **self.tokenizer_decode_config)
        
        if self.return_states:
            select_states = {'past_key_values': states['past_key_values'], 
                             'last_token_id': output_ids[0].cpu().numpy()[-1],
                             'last_token': self.tokenizer.decode(output_ids[0].cpu().numpy()[-1])}
            return output_text, select_states
        else:
            return output_text

    @inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def generate(self, input_ids:Union[str, list, torch.Tensor], **kwargs):
        '''单条样本生成 / batch生成'''
        if isinstance(input_ids, str):
            # 单条样本
            self.use_batch = False
        elif isinstance(input_ids, list):
            # batch生成
            self.use_batch = True
            # 为多个query同时生成时候，仅允许为每条query生成1条response，即n=1
            if 'generation_config' in kwargs:
                if kwargs['generation_config'].get('n', 0) > 1:
                    log_warn_once(f"Args `generation_config['n']`={kwargs['n']} has been reset to `1` when batch generate")
                kwargs['generation_config']['n'] = 1
            else:
                if kwargs.get('n', 0) > 1:
                    log_warn_once(f"Args `n`={kwargs['n']} has been reset to `1` when batch generate")
                kwargs['n'] = 1
            if self.use_states and (self.padding_side == 'right') and self.encoder is None:
                # 对于decoder_only的模型,非encoder_decoder的, batch推理智能padding在左侧
                self.padding_side = 'left'
                log_info("When arg `use_states`=True, reset `padding_side`='left' to avoid error output")
        elif isinstance(input_ids, torch.Tensor):
            self.use_batch = False if input_ids.shape[0] == 1 else True
        else:
            raise TypeError('Args `text` only support `str/list(str)` format')
        
        self.set_generation_config(kwargs)
        inputs = self.pre_process(input_ids)

        if self.mode == 'random_sample':
            output_ids = self.random_sample(inputs, states=self.decoder._get_initial_model_kwargs(kwargs))  # 基于随机采样
        elif self.mode == 'beam_search':
            output_ids = self.beam_search(inputs, states=self.decoder._get_initial_model_kwargs(kwargs))  # 基于beam search

        return self.post_process(output_ids)

    @inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def stream_generate(self, input_ids:Union[str, torch.Tensor], **kwargs):
        '''单条样本stream输出预测的结果'''
        self.set_generation_config(kwargs)
        self.use_batch = False
        inputs = self.pre_process(input_ids)
        if self.mode == 'random_sample':
            for output_ids in self.stream_random_sample(inputs, states=self.decoder._get_initial_model_kwargs(kwargs)):  # stream随机采样
                yield self.post_process(output_ids)
        elif self.mode == 'beam_search':
            for output_ids in self.stream_beam_search(inputs, states=self.decoder._get_initial_model_kwargs(kwargs)):  # stream beam采样
                yield self.post_process(output_ids)


class Seq2SeqGeneration(SeqGeneration):
    '''encoder-decoder语言模型的解码'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.decoder.encoder
        self.decoder = self.decoder.decoder
        self.use_segment_ids = hasattr(self.encoder, 'segment_vocab_size') and (self.encoder.segment_vocab_size > 0)  # 是否使用segment_ids

    def _prepare_next_inputs(self, encoder_outputs, decoder_inputs, include_past=True, **states):
        if include_past is False:
            # 这里未做判断, 因为一般seq2seq模型都没有segment_ids
            inputs = [decoder_inputs[:, -1:]] + encoder_outputs
        else:
            inputs = [decoder_inputs] + encoder_outputs
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        self.input_seqlen = torch.zeros(decoder_inputs.shape[0], dtype=torch.long).to(self.device)
        states = self.decoder.prepare_inputs_for_generation(*inputs, **states)
        return inputs, states

    def pre_process(self, text:Union[str, torch.Tensor, List[Union[int, str, torch.Tensor]], Tuple[Union[int, str, torch.Tensor]]]) -> \
                    Union[torch.Tensor, List[Union[torch.Tensor, List[int]]]]:
        inputs = super().pre_process(text)
        inputs = self._trans2tensors(inputs)
        encoder_output = self.encoder.predict(inputs)  # 在pre_process阶段直接过完encoder
        return encoder_output
    