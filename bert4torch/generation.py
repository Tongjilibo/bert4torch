'''自回归模型的生成
'''

import torch
import torch.nn as nn
import numpy as np
import inspect
from bert4torch.snippets import take_along_dim, torch_div, sequence_padding, create_position_ids_start_at_padding, log_info, log_warn
from bert4torch.tokenizers import TokenizerBase
from packaging import version
from contextlib import contextmanager
import gc


if version.parse(torch.__version__) >= version.parse("1.10.0"):
    model_inference_mode = torch.inference_mode
else:
    model_inference_mode = torch.no_grad

def repetition_penalty_func(input_ids: torch.LongTensor, scores: torch.FloatTensor, penalty: float) -> torch.FloatTensor:
    score = torch.gather(scores, 1, input_ids)
    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)
    scores.scatter_(1, input_ids, score)
    return scores


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

    :param start_id: int, 解码使用的起始token_id，不同预训练模型设置可能不一样
    :param end_id: int/tuple/list, 解码使用的结束token_id，不同预训练模型设置可能不一样, 默认给的-1（真实场景中不存在，表示输出到maxlen）
    :param maxlen: int, 最大解码长度
    :param minlen: int, 最小解码长度, 默认为1
    :param pad_id: int, pad_id，在batch解码时候使用
    :param pad_mode: str, padding在前面还是后面，pre或者post
    :param device: str, 默认为'cpu'
    :param n: int, random_sample时候表示生成的个数；beam_search时表示束宽
    :param topk: int, 这里的topk是指仅保留topk的值
    :param topp: float, 这里的topp是token的概率阈值设置
    :param temperature: 温度参数，默认为1
    :param repetition_penalty: 重复的惩罚系数
    :param min_ends: int, 最小的end_id的个数
    """
    @model_inference_mode()
    def __init__(self, start_id=None, end_id=-1, maxlen=64, minlen=1, pad_id=0, pad_mode='post', device='cpu', 
                 n=1, topk=50, topp=1, temperature=1, repetition_penalty=1.0, min_ends=1, **generation_config):
        # generation_config
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.pad_id = pad_id   # pad_token_id兼容bert4torch和hf的，如错误则需要显式传入pad_id:int
        self.pad_mode = pad_mode
        self.device = device
        self.n = n
        self.topk = topk
        self.topp = topp
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.min_ends = min_ends
        # 参数别名：兼容transformers的参数设置
        self.alias = {'bos_token_id': 'start_id',
                      'eos_token_id': 'end_id',
                      'top_k': 'topk',
                      'top_p': 'topp',
                      'max_new_tokens': 'maxlen',
                      'min_new_tokens': 'minlen',
                      'pad_token_id': 'pad_id'
                     }
        self.set_generation_config(generation_config)
                
        self.use_batch = False
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    def set_generation_config(self, generation_config):
        for key, value in generation_config.items():
            if key in self.alias:  # 兼容transformers的参数设置
                setattr(self, self.alias[key], value)
            elif hasattr(self, key):
                setattr(self, key, value)  # 在generate()时候动态设置属性
            # else:
            #     log_warn_once(f'Generation_config `{key}` has not been pre maintained')

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数

        目前包含: 
            1. 设置rtype参数，并做相应处理；
            2. 确定states的使用，并做相应处理；
            3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(self, inputs, output_ids, states, rtype=default_rtype):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if use_states:
                    assert len(prediction) == 2, 'Should return 2 output when set use_states=True'
                else:
                    prediction = (prediction, None)

                # repetition_penalty
                if self.repetition_penalty != 1.0:
                    past_token_ids = inputs[0] if states is None else states['past_token_ids']
                    prediction = (repetition_penalty_func(past_token_ids, prediction[0], self.repetition_penalty), prediction[1])

                if default_rtype == 'logits':
                    prediction = (nn.Softmax(dim=-1)(prediction[0] / self.temperature), prediction[1])
                elif self.temperature != 1:
                    probas = torch.pow(prediction[0], 1.0 / self.temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return torch.log(prediction[0] + 1e-12), prediction[1]

            # 增加函数，用于动态修改闭包中的属性
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

    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数；
        说明: 定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
        rtype=logits时则返回softmax前的结果或者概率对数。
        
        :return: 二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def _prepare_raw_inputs(self, inputs_raw) -> list:
        '''对当前输入进行处理, 并都转化成tensor, return: list[tensor]'''
        # 传入的Tensor直接[]后返回
        if isinstance(inputs_raw, torch.Tensor):
            self.input_seqlen = torch.ones(inputs_raw.shape[0], dtype=torch.long).to(self.device) * inputs_raw.shape[1]
            return [inputs_raw.to(self.device)]
        inputs = []
        for input_ in inputs_raw:
            # encoder-decoder传入的encoder_hidden_states和encoder_attention_mask
            if isinstance(input_, torch.Tensor):
                input_new = input_
            elif isinstance(input_, (list, tuple, np.ndarray)):
                # 单条样本为[1,2,3]格式，需转为[[1,2,3]]
                input_ = input_ if self.use_batch else [input_]
                input_new = torch.tensor(sequence_padding(input_, value=self.pad_id, mode=self.pad_mode), dtype=torch.long, device=self.device)

                # padding在右边则input_seqlen是真实的长度，左边则统一为最大程度
                if self.pad_mode in {'post', 'right'}:
                    self.input_seqlen = torch.tensor([len(i) for i in input_], dtype=torch.long).to(self.device)
                else:
                    max_len = input_new.shape[1]
                    self.input_seqlen = torch.tensor([max_len]*len(input_new), dtype=torch.long).to(self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(input_new)
        return inputs

    def _identify_sentence_end(self, output_ids):
        '''判断句子是否结束'''
        if isinstance(self.end_id, (int, float)):
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        elif isinstance(self.end_id, (set, tuple, list)):
            end_counts = 0
            is_end = None
            for end_id in self.end_id:
                tmp = output_ids[:, -1] == end_id
                is_end = tmp if is_end is None else is_end + tmp
                end_counts += (output_ids == end_id).sum(1)
        return is_end, end_counts

    def __beam_search_step(self, step, inputs, output_ids, output_scores, states):
        '''beam_search单条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, 'logits')  # 计算当前得分
        if step == 0:  # 第1步预测后将输入重复topk次
            inputs = [i.repeat([self.topk]+[1]*(len(i.shape)-1)) for i in inputs]
        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        indices = scores.flatten().argsort(dim=-1, descending=True)[:self.topk]  # 仅保留topk
        indices_1 = torch_div(indices, scores.shape[1], rounding_mode='floor')  # 兼容老版本
        indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
        output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
        output_scores = take_along_dim(scores, indices, dim=None)  # 更新得分
        return inputs, output_ids, output_scores, states

    def __beam_search_end(self, inputs, output_ids, output_scores, results):
        '''beam_search单条推理计算是否结束'''
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        flag = ~is_end | (end_counts < self.min_ends)  # 标记未完成序列
        self.flag = flag  # 记录未完成序列
        if output_ids.shape[1] >= self.minlen:  # 最短长度判断
            best = output_scores.argmax()  # 得分最大的那个
            if is_end[best] and end_counts[best] >= self.min_ends:  # 如果已经终止
                break_tag = True
            elif not flag.all():  # 如果有已完成的
                inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                output_ids = output_ids[flag]  # 扔掉已完成序列
                output_scores = output_scores[flag]  # 扔掉已完成序列
                end_counts = end_counts[flag]  # 扔掉已完成end计数
                self.topk = flag.sum()  # topk相应变化
        return inputs, output_ids, output_scores, results, break_tag

    def __batch_beam_search_step(self, step, inputs, output_ids, output_scores, states):
        '''beam_search batch条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, 'logits')  # 计算当前得分
        if step == 0:  # 第0步预测后将输入重复topk次
            inputs_new = []
            for input_ in inputs:
                inputs_ = []
                for topk, input_i in zip(self.topk, input_):
                    input_i = input_i.unsqueeze(0)
                    inputs_.append(input_i.repeat([topk]+[1]*(len(input_i.shape)-1)))
                inputs_new.append(torch.cat(inputs_))
            inputs = inputs_new
            # 对seq_len进行扩充
            input_seqlen = []
            for topk, input_seqlen_i in zip(self.topk, self.input_seqlen):
                input_seqlen.append(input_seqlen_i.repeat(topk))
            self.input_seqlen = torch.cat(input_seqlen)

        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        output_ids_new, output_scores_new = [], []
        for smp_i, topk in enumerate(self.topk):
            if step == 0:
                score = scores[smp_i][None, ...]
                output_id = output_ids[smp_i][None, ...]
            else:
                start, end = sum(self.topk[:smp_i]), sum(self.topk[:smp_i+1])
                score = scores[start:end]
                output_id = output_ids[start:end]

            indices = score.flatten().argsort(dim=-1, descending=True)[:topk]  # 仅保留topk
            indices_1 = torch_div(indices, score.shape[1], rounding_mode='floor')  # 兼容老版本
            indices_2 = (indices % score.shape[1]).reshape((-1, 1))  # 列索引
            output_id = torch.cat([output_id[indices_1], indices_2], 1)  # 更新输出
            output_score = take_along_dim(score, indices, dim=None)  # 更新得分
            output_ids_new.append(output_id)
            output_scores_new.append(output_score)
        output_ids_new = torch.cat(output_ids_new)
        output_scores_new = torch.cat(output_scores_new)
        return inputs, output_ids_new, output_scores_new, states

    def __batch_beam_search_end(self, inputs, output_ids, output_scores, results):
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        self.flag = ~is_end | (end_counts < self.min_ends)  # 标记未完成序列

        if output_ids.shape[1] >= self.minlen:  # 最短长度判断
            inputs_new, output_ids_new, output_scores_new, flag_new = [], [], [], []
            topks_new = self.topk.copy()
            for smp_i, topk in enumerate(self.topk): # 这里的topk是一个list
                if topk == 0:
                    continue
                start, end = sum(self.topk[:smp_i]), sum(self.topk[:smp_i+1])
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
            self.topk = topks_new
            self.flag = torch.cat(flag_new)

            if len(output_ids) == 0:
                break_tag = True
        else:
            # 不满足结束条件，需要对self.flag进行更新
            self.flag = torch.ones_like(self.flag, dtype=torch.bool)
        
        return inputs, output_ids, output_scores, results, break_tag

    def beam_search(self, inputs, states=None, **generation_config):
        """beam search解码
        
        :param inputs: 编码器的输入，包含encoder_hidden_states, encoder_attention_mask
        :param topk: int, 这里的topk即beam size
        :param states:
        :param temperature: 温度参数，默认为1
        :param min_ends:
        :return: 最优解码序列。
        """
        self.set_generation_config(generation_config)
        assert self.topk is not None, 'Arg `topk` means beam_size anc can not be None'
        inputs = self._prepare_raw_inputs(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        output_scores = torch.zeros(btz, device=self.device)
        results = []

        # batch推理
        if self.use_batch:
            self.topk = [self.topk] * btz
            results = [None] * btz

        for step in range(self.maxlen):
            if (not self.use_batch):  # 单条推理
                inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, states)
                inputs, output_ids, output_scores, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, results)
            else:  # batch推理
                inputs, output_ids, output_scores, states = self.__batch_beam_search_step(step, inputs, output_ids, output_scores, states)
                inputs, output_ids, output_scores, results, break_tag = self.__batch_beam_search_end(inputs, output_ids, output_scores, results)
            if break_tag:
                break 
        # 如果还有未完成序列，直接放入结果
        if len(output_ids) > 0:
            if not self.use_batch:
                results.append(output_ids[output_scores.argmax()])
            elif self.use_batch:
                for smp_i, result in enumerate(results):
                    if result is None:
                        start, end = sum(self.topk[:smp_i]), sum(self.topk[:smp_i+1])
                        output_score = output_scores[start:end]
                        output_id = output_ids[start:end]
                        results[smp_i] = output_id[output_score.argmax()]

        # 达到长度直接输出
        self.flag = None
        return results

    def stream_beam_search(self, inputs, states=None, **generation_config):
        '''beam_search的stream输出模式'''
        self.set_generation_config(generation_config)
        assert self.topk is not None, 'Arg `topk` means beam_size anc can not be None'
        inputs = self._prepare_raw_inputs(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        output_scores = torch.zeros(btz, device=self.device)
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, states)
            yield [output_ids[output_scores.argmax()]]
            inputs, output_ids, output_scores, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, results)
            if break_tag:
                break 
            
        # 达到长度直接输出
        self.flag = None

    def __random_sample_step(self, step, inputs, output_ids, states):
        '''为了random_sample和stream_random_sample共用，抽离出来的单步逻辑'''
        self.step = step
        probas, states = self.predict(inputs, output_ids, states, 'probas')  # 计算当前概率
        probas /= probas.sum(dim=-1, keepdims=True)  # 确保归一化
        if step == 0:  # 第1步预测后将结果重复n次
            probas = probas.repeat([self.n]+[1]*(len(probas.shape)-1))
            inputs = [i.repeat([self.n]+[1]*(len(i.shape)-1)) for i in inputs]
            output_ids = output_ids.repeat([self.n]+[1]*(len(output_ids.shape)-1))
        if self.topk is not None:
            k_indices = probas.argsort(dim=-1, descending=True)[:, :self.topk]  # 仅保留topk
            probas = take_along_dim(probas, k_indices, dim=1)  # topk概率
            probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化
        if self.topp is not None:
            p_indices = probas.argsort(dim=-1, descending=True)  # 从高到低排序
            probas = take_along_dim(probas, p_indices, dim=-1)  # 排序概率
            cumsum_probas = torch.cumsum(probas, dim=-1)  # 累积概率
            flag = torch.roll(cumsum_probas >= self.topp, 1, dims=1)  # 标记超过topp的部分
            flag[:, 0] = False  # 结合上面的torch.roll，实现平移一位的效果
            probas[flag] = 0  # 后面的全部置零
            probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化

        sample_func = lambda p: torch.multinomial(p, 1)  # 按概率采样函数
        sample_ids = torch.stack([sample_func(p) for p in probas])
        sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
        if self.topp is not None:
            sample_ids = take_along_dim(p_indices, sample_ids, dim=1)  # 对齐原id
        if self.topk is not None:
            sample_ids = take_along_dim(k_indices, sample_ids, dim=1)  # 对齐原id
        output_ids = torch.cat([output_ids, sample_ids], 1)  # 更新输出
        return inputs, output_ids, states

    def __random_sample_end(self, inputs, output_ids, results, smp_indexs=None):
        break_tag = False
        is_end, end_counts = self._identify_sentence_end(output_ids)
        f_flag = is_end & (end_counts >= self.min_ends)  # 标记已完成序列
        self.flag = (f_flag == False)  # 记录未完成序列，这里是裁前的，用于use_states时候的裁剪操作
        if (output_ids.shape[1] >= self.minlen) and f_flag.any():  # 最短长度判断, 如果有已完成的
            if smp_indexs is None:  # single
                for ids in output_ids[f_flag]:  # 存好已完成序列
                    results.append(ids)
            else:  # batch
                for smp_i, ids in zip(smp_indexs[f_flag], output_ids[f_flag]):  # 存好已完成序列
                    results[smp_i] = ids

            flag = (f_flag == False)  # 标记未完成序列，True表示未完成
            inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
            output_ids = output_ids[flag]  # 只保留未完成部分候选集
            # end_counts = end_counts[flag]  # 只保留未完成部分end计数
            if smp_indexs is not None:
                smp_indexs = smp_indexs[flag]
            if len(output_ids) == 0:
                break_tag = True
        else:
            # 不满足结束条件，需要对self.flag进行更新
            self.flag = torch.ones_like(self.flag, dtype=torch.bool)

        if smp_indexs is None:
            return inputs, output_ids, results, break_tag
        else:
            return inputs, output_ids, results, break_tag, smp_indexs

    def random_sample(self, inputs_raw, states=None, **generation_config):
        """随机采样n个结果；
        说明: 非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        
        :param inputs_raw: tensor、array、list、tuple, 解码的输入，一般为last_hidden_state, shape=[btz, seq_len, hdsz]
        :param topk: int, 这里的topk是指仅保留topk的值
        :param topp: float, 这里的topp是token的概率阈值设置
        :param states: None/dict, 缓存参数
        :param temperature: 温度参数，默认为1
        :param min_ends: 最小的end_id的个数
        :return: n个解码序列组成的list。
        """
        self.set_generation_config(generation_config)
        inputs = self._prepare_raw_inputs(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        results = []

        if self.use_batch:
            smp_indexs = torch.tensor(range(btz)).to(output_ids)
            results = [None] * btz

        for step in range(self.maxlen):
            if not self.use_batch:  # single
                inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)
                inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results)
            else:  # batch
                inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)
                inputs, output_ids, results, break_tag, smp_indexs = self.__random_sample_end(inputs, output_ids, results, smp_indexs)
            if break_tag:
                break

        # 如果还有未完成序列，直接放入结果
        if len(output_ids) > 0:
            if not self.use_batch:
                for ids in output_ids:
                    results.append(ids)
            elif self.use_batch:
                for smp_i, ids in zip(smp_indexs, output_ids):  # 存好已完成序列
                    results[smp_i] = ids
        # 返回结果
        self.flag = None
        return results
    
    def stream_random_sample(self, inputs_raw, states=None, **generation_config):
        """随机采样n个结果；stream输出"""
        generation_config['n'] = 1
        self.set_generation_config(generation_config)
        inputs = self._prepare_raw_inputs(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, states = self.__random_sample_step(step, inputs, output_ids, states)
            yield output_ids
            inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results)
            if break_tag:
                break
        self.flag = None


class SeqGeneration(AutoRegressiveDecoder):
    '''单向decoder语言模型的解码，对AutoRegressiveDecoder的简单封装，可以使用cache来加快解码

    :param model: 模型
    :param tokenizer: tokenizer，如果使用第三方的tokenize，需要继承重写下pre_process和post_process
    :param tokenizer_config: dict, tokenize的参数, 一般设置这个即可，若部分参数无法设置，则单独设置tokenizer_encode_config和tokenizer_decode_config
    :param tokenizer_encode_config: dict, tokenize的encode参数
    :param tokenizer_decode_config: dict, tokenize的decode参数
    :param default_rtype: str, 模型输出的结果是logits设置为logits，如果是probas设置为probas
    :param use_states: str, 是否使用cache
    :param device: str, 默认为None，因为可以直接使用传入的model.device

    > generation_config
    :param start_id: int, 解码使用的起始token_id，不同预训练模型设置可能不一样
    :param end_id: int/tuple/list, 解码使用的结束token_id，不同预训练模型设置可能不一样, 默认给的-1（真实场景中不存在，表示输出到maxlen）
    :param maxlen: int, 最大解码长度
    :param minlen: int, 最小解码长度, 默认为1
    :param pad_id: int, pad_id，在batch解码时候使用
    :param pad_mode: str, padding在前面还是后面，pre或者post
    :param device: str, 默认为'cpu'
    :param n: int, random_sample时候表示生成的个数；beam_search时表示束宽
    :param topk: int, 这里的topk是指仅保留topk的值
    :param topp: float, 这里的topp是token的概率阈值设置
    :param temperature: 温度参数，默认为1
    :param repetition_penalty: 重复的惩罚系数
    :param min_ends: int, 最小的end_id的个数
    '''
    def __init__(self, model, tokenizer=None, tokenizer_config=None, tokenizer_encode_config=None, tokenizer_decode_config=None, 
                 mode='random_sample', default_rtype='logits', use_states=True, optimize_cuda_cache=False, **generation_config):
        generation_config = self._default_generation_config(tokenizer, model, generation_config)  # 对部分参数进行默认设置
        super().__init__(**generation_config)

        self.encoder = None
        self.decoder = model

        # tokenizer参数
        self.tokenizer = tokenizer
        self.tokenizer_type = 'b4t' if isinstance(tokenizer, TokenizerBase) else 'hf'
        tokenizer_config = tokenizer_config or dict()
        tokenizer_config.update({'maxlen': self.maxlen})
        tokenizer_encode_config = tokenizer_encode_config or {}
        tokenizer_decode_config = tokenizer_decode_config or {}
        if self.tokenizer is not None:
            self.tokenizer_encode_config = {**self.clear_tokenizer_config(tokenizer_config, self.tokenizer.encode), **tokenizer_encode_config}
            self.tokenizer_decode_config = {**self.clear_tokenizer_config(tokenizer_config, self.tokenizer.decode), **tokenizer_decode_config}

        assert mode in {'random_sample', 'beam_search'}, 'Args `mode` only support `random_sample/beam_search`.'
        self.mode = mode
        self.predict.set_default_rtype(default_rtype)  # 动态修改闭包中的default_rtype
        self.predict.set_use_states(use_states)  # 动态修改闭包中的use_states
        self.use_states = use_states
        self.use_segment_ids = hasattr(model, 'segment_vocab_size') and (model.segment_vocab_size > 0)  # 是否使用segment_ids
        self.include_input = False  # 输出是否包含输入
        self.input_text = ''
        EmptyCacheDecorators.optimize_cuda_cache = optimize_cuda_cache
    
    @staticmethod
    def _default_generation_config(tokenizer, model, generation_config):
        ''' genration的默认参数设置 '''
        if generation_config.get('pad_id') is not None:  # 用户自行设置
            pass
        elif generation_config.get('pad_token_id') is not None:  # 用户自行设置（别名）
            pass
        elif (tokenizer is not None) and (tokenizer.pad_token_id is not None):
            generation_config['pad_id'] = tokenizer.pad_token_id
        else:
            generation_config['pad_id'] = 0
            log_info(f'Arg `pad_id` has been set to default value 0')
        
        if generation_config.get('end_id') is not None:  # 用户自行设置
            pass
        elif generation_config.get('eos_token_id') is not None:  # 用户自行设置（别名）
            pass
        elif (tokenizer is not None) and hasattr(tokenizer, 'eos_token_id') and (tokenizer.eos_token_id is not None):
            generation_config['end_id'] = tokenizer.eos_token_id
            log_info(f'Arg `end_id` has been set to tokenizer.eos_token_id:{tokenizer.eos_token_id}')
        else:
            generation_config['end_id'] = generation_config['pad_id']
            log_info(f'Arg `end_id` has been set to `pad_id`:{generation_config["pad_id"]}')
        generation_config['device'] = generation_config.get('device') or next(model.parameters()).device
        return generation_config

    def _prepare_next_inputs(self, inputs, output_ids, include_past=True):
        '''decode时候准备下一次输入，使用cache则仅输入last_token_ids，不适用需要输入past_token_ids'''
        def concat_token_ids(token_ids, output_ids):
            '''把非padding部分concat在一起
            '''
            if not self.use_batch:
                return torch.cat([token_ids, output_ids], 1)
            
            # 部分序列已经完成，需要更新input_seqlen
            if len(self.input_seqlen) != len(output_ids):
                self.input_seqlen = self.input_seqlen[self.flag]
            inputs = []
            for seq_l, token_ids_i, output_ids_i in zip(self.input_seqlen, token_ids, output_ids):
                inputs.append(torch.cat([token_ids_i[:seq_l], output_ids_i, token_ids_i[seq_l:]]))
            return torch.stack(inputs)

        if include_past is False:
            if len(inputs) == 1:
                next_inputs = [output_ids[:, -1:]]
            elif len(inputs) >= 2:  # 第二个是segment_ids
                token_ids = inputs[0]
                curr_segment_ids = token_ids[0, -1:]
                next_inputs = [output_ids[:, -1:], curr_segment_ids]
        else:
            if len(inputs) == 1:
                token_ids = concat_token_ids(inputs[0], output_ids)
                next_inputs = [token_ids]
            elif len(inputs) >= 2:  # 第二个是segment_ids
                token_ids, segment_ids = inputs
                token_ids = concat_token_ids(inputs[0], output_ids)
                curr_segment_ids = torch.zeros_like(output_ids) + token_ids[0, -1]
                segment_ids = torch.cat([segment_ids, curr_segment_ids], 1)
                next_inputs = [token_ids, segment_ids]
        return next_inputs

    def __get_last_token_logits(self, logits, output_ids):
        '''获取最后一个token的logits'''
        if not self.use_batch:
            return logits[:, -1, :]
        else:
            # 由于batch是padding过的，因此要找到padding前的位置
            last_token_index = self.input_seqlen + output_ids.shape[1] - 1
            return torch.stack([logit[index_, :] for index_, logit in zip(last_token_index, logits)])

    @AutoRegressiveDecoder.wraps()
    def predict(self, inputs, output_ids, states):
        assert isinstance(inputs, (tuple, list))
        if states is not None:
            assert self.use_states is True, 'Args `use_states` must be True when return states is not None'
        
        # 使用cache, 输入只能padding在左侧
        if self.use_states:
            states = {'use_states': True} if states is None else states

            # next_inputs：step=0时候输入全部，>=1时候输入last_token
            next_inputs = self._prepare_next_inputs(inputs, output_ids, include_past=self.step==0)

            # past_token_ids: inputs+output_ids
            if self.step >= 1:
                states['past_token_ids'] = self._prepare_next_inputs(inputs, output_ids)[0]
            
            # position_ids: 在第0步如果padding在左侧，则需要自定义position_ids, >=1步则取last_token
            if self.use_batch and (self.step==0) and (self.pad_mode in {'pre', 'left'}) :
                # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                #         [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])
                states['position_ids'] = create_position_ids_start_at_padding(next_inputs[0], self.pad_id, past_key_values_length=-1, start_padding_idx=False)
            elif states.get('position_ids') is not None:
                states['position_ids'] = states['position_ids'][:, -1:] + 1

            # attention_mask: 根据token_ids生成的，因此这里重置下
            if states.get('input_attention_mask') is not None:  # 在states中input_attention_mask才是[btz, seq_len]
                attention_mask = states['input_attention_mask']
                states['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                del states['input_attention_mask']

            # shape和size不一致: step=1时候，beam_search的btz=束宽, 或者要返回多个结果
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

            # 已经有序列完成，仅保留未完成序列
            if hasattr(self, 'flag') and (self.flag is not None):
                states['attention_mask'] = states['attention_mask'][self.flag]
                if (len(states['position_ids']) > 1):
                    states['position_ids'] = states['position_ids'][self.flag]
                for l_i, past_key_value in enumerate(states['past_key_values']):
                    states['past_key_values'][l_i] = (past_key_value[0][self.flag], past_key_value[1][self.flag])
                if 'cross_past_key_values' in states:
                    for l_i, cross_past_key_value in enumerate(states['cross_past_key_values']):
                        states['cross_past_key_values'][l_i] = (cross_past_key_value[0][self.flag], cross_past_key_value[1][self.flag])

            logits, states = self.decoder.predict(next_inputs, **states)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return logits[:, -1, :], states

        # 不使用cache
        elif not self.use_states:
            next_inputs = self._prepare_next_inputs(inputs, output_ids, include_past=True)

            # 如果use_states=False且pad_mode='pre'，则需要自定义position_ids，否则position_ids不正确，但这么使用很少
            position_ids = None
            if self.pad_mode in {'pre', 'left'}:
                position_ids = create_position_ids_start_at_padding(next_inputs[0], self.pad_id, past_key_values_length=-1, start_padding_idx=False)
                logits = self.decoder.predict(next_inputs, position_ids=position_ids)
            else:
                logits = self.decoder.predict(next_inputs)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return self.__get_last_token_logits(logits, output_ids)

    @staticmethod
    def clear_tokenizer_config(config, func):
        '''获取在tokenize_params中的参数'''
        arg_names = [k for k in inspect.signature(func).parameters]
        return {k:v for k, v in config.items() if k in arg_names}

    def pre_process(self, text):
        '''前处理，可以继承后自定义，主要用于第三方tokenizer的encode'''
        self.input_text = text if self.include_input else ''
        if self.tokenizer is None:
            return text
        
        if self.tokenizer_type == 'b4t':
            # bert4torch的tokenizer
            inputs = self.tokenizer.encode(text, **self.tokenizer_encode_config)
            return inputs if self.use_segment_ids else [inputs[0]]
        elif self.tokenizer_type == 'hf':
            # hf的tokenize
            return [self.tokenizer(text, **self.tokenizer_encode_config)['input_ids']]
    
    def post_process(self, output_ids):
        '''后处理，可以继承后自定义，主要用于第三方tokenizer的decode'''
        if self.tokenizer is None:
            return output_ids

        if len(output_ids) > 1:
            outputs = [self.tokenizer.decode(ids.cpu().numpy(), **self.tokenizer_decode_config) for ids in output_ids]
            if isinstance(self.input_text, str):
                # 输入是str，则在前面直接添加
                return [self.input_text + item for item in outputs]
            elif isinstance(self.input_text, (tuple, list)) and (len(self.input_text) == len(outputs)):
                # 输入是list且和outputs同维度
                return [self.input_text[i] + item for i, item in enumerate(outputs)]
            return outputs
        elif len(output_ids) == 1:
            return self.input_text + self.tokenizer.decode(output_ids[0].cpu().numpy(), **self.tokenizer_decode_config)
        return output_ids

    def _generate(self, inputs):
        if self.mode == 'random_sample':
            output_ids = self.random_sample(inputs)  # 基于随机采样
        elif self.mode == 'beam_search':
            output_ids = self.beam_search(inputs)  # 基于beam search
        return output_ids

    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def generate(self, text:str, **generation_config):
        '''单条样本生成'''
        self.set_generation_config(generation_config)
        self.use_batch = False
        inputs = self.pre_process(text)
        output_ids = self._generate(inputs)
        return self.post_process(output_ids)

    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def batch_generate(self, text_list:list, **generation_config):
        '''batch样本生成，use_states=True时要求pad_mode='pre', use_states=False时候对'''
        # 参数设定
        generation_config['n'] = 1
        self.set_generation_config(generation_config)
        self.use_batch = True
        if self.use_states and (self.pad_mode in {'post', 'right'}):
            self.pad_mode = 'pre'
            log_info("When arg `use_states`=True, you may set `pad_mode`='pre' to avoid error output, reset `pad_mode`='pre' instead")

        # 主流程
        inputs = self.pre_process(text_list)
        output_ids = self._generate(inputs)
        return self.post_process(output_ids)

    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def stream_generate(self, text:str, **generation_config):
        '''单条样本stream输出预测的结果'''
        self.set_generation_config(generation_config)
        self.use_batch = False
        inputs = self.pre_process(text)
        if self.mode == 'random_sample':
            for output_ids in  self.stream_random_sample(inputs):  # stream随机采样
                yield self.post_process(output_ids)
        elif self.mode == 'beam_search':
            for output_ids in  self.stream_beam_search(inputs):  # stream随机采样
                yield self.post_process(output_ids)


class Seq2SeqGeneration(SeqGeneration):
    '''encoder-decoder语言模型的解码'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.decoder.encoder
        self.decoder = self.decoder.decoder
        self.use_segment_ids = hasattr(self.encoder, 'segment_vocab_size') and (self.encoder.segment_vocab_size > 0)  # 是否使用segment_ids

    def _prepare_next_inputs(self, encoder_outputs, decoder_inputs, include_past=True):
        if include_past is False:
            # 这里未做判断，因为一般seq2seq模型都没有segment_ids
            inputs = [decoder_inputs[:, -1:]] + encoder_outputs
        else:
            inputs = [decoder_inputs] + encoder_outputs
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        self.input_seqlen = torch.zeros(decoder_inputs.shape[0], dtype=torch.long).to(self.device)
        return inputs
    
    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def generate(self, text:str, **generation_config):
        self.set_generation_config(generation_config)
        self.use_batch = False
        inputs = self.pre_process(text)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        output_ids = super()._generate(encoder_output)
        return self.post_process(output_ids)

    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def batch_generate(self, text_list:list, **generation_config):
        '''batch样本生成'''
        # 参数设定
        generation_config['n'] = 1
        self.set_generation_config(generation_config)
        self.use_batch = True
        inputs = self.pre_process(text_list)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        output_ids = super()._generate(encoder_output)
        return self.post_process(output_ids)

    @model_inference_mode()
    @EmptyCacheDecorators.empty_cuda_cache()
    def stream_generate(self, text:str, **generation_config):
        '''stream输出t预测的结果'''
        self.set_generation_config(generation_config)

        self.use_batch = False
        inputs = self.pre_process(text)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        if self.mode == 'random_sample':
            for output_ids in  self.stream_random_sample(encoder_output):  # stream随机采样
                yield self.post_process(output_ids)
        elif self.mode == 'beam_search':
            for output_ids in  self.stream_beam_search(encoder_output):  # stream随机采样
                yield self.post_process(output_ids)
