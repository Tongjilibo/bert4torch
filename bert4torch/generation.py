'''自回归模型的生成
'''

import torch
import torch.nn as nn
import numpy as np
from bert4torch.snippets import take_along_dim, torch_div, sequence_padding, create_position_ids_start_at_padding

class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略

    :param start_id: int, 解码使用的起始token_id，不同预训练模型设置可能不一样
    :param end_id: int, 解码使用的结束token_id，不同预训练模型设置可能不一样
    :param maxlen: int, 最大解码长度
    :param minlen: int, 最小解码长度, 默认为1
    :param device: str, 默认为'cpu'
    """
    def __init__(self, start_id, end_id, maxlen, minlen=1, pad_id=0, pad_mode='post', device='cpu'):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.device = device
        self.use_batch = False
        self.pad_id = pad_id   # pad_token_id兼容bert4torch和hf的，如错误则需要显式传入pad_id:int
        self.pad_mode = pad_mode
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数

        目前包含: 
            1. 设置rtype参数，并做相应处理；
            2. 确定states的使用，并做相应处理；
            3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(self, inputs, output_ids, states, temperature=1, rtype=default_rtype):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if use_states:
                    assert len(prediction) == 2, 'Should return 2 output when set use_states=True'
                else:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (nn.Softmax(dim=-1)(prediction[0] / temperature), prediction[1])
                elif temperature != 1:
                    probas = torch.pow(prediction[0], 1.0 / temperature)
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
            self.input_seqlen = torch.ones(inputs_raw.shape[0]).to(self.device) * inputs_raw.shape[1]
            return [inputs_raw.to(self.device)]
        inputs = []
        for input_ in inputs_raw:
            # encoder-decoder传入的encoder_hidden_states和encoder_attention_mask
            if isinstance(input_, torch.Tensor):
                input_new = input_
            elif isinstance(input_, (list, tuple, np.ndarray)):
                # 单条样本为[1,2,3]格式，需转为[[1,2,3]]
                input_ = input_ if self.use_batch else [input_]
                input_new = torch.tensor(sequence_padding(input_, value=self.pad_id, mode=self.pad_mode), device=self.device)

                # padding在右边则input_seqlen是真实的长度，左边则统一为最大程度
                if self.pad_mode in {'post', 'right'}:
                    self.input_seqlen = torch.tensor([len(i) for i in input_]).to(self.device)
                else:
                    max_len = input_new.shape[1]
                    self.input_seqlen = torch.tensor([max_len]*len(input_new)).to(self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(input_new)
        return inputs

    def __beam_search_step(self, step, inputs, output_ids, output_scores, topk, states, temperature):
        '''beam_search单条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, temperature, 'logits')  # 计算当前得分
        if step == 0:  # 第1步预测后将输入重复topk次
            inputs = [i.repeat([topk]+[1]*(len(i.shape)-1)) for i in inputs]
        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        indices = scores.flatten().argsort(dim=-1, descending=True)[:topk]  # 仅保留topk
        indices_1 = torch_div(indices, scores.shape[1], rounding_mode='floor')  # 兼容老版本
        indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
        output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
        output_scores = take_along_dim(scores, indices, dim=None)  # 更新得分
        return inputs, output_ids, output_scores, states

    def __beam_search_end(self, inputs, output_ids, output_scores, topk, results, min_ends):
        '''beam_search单条推理计算是否结束'''
        break_tag = False
        is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
        end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
        self.flag = flag  # 记录未完成序列
        if output_ids.shape[1] >= self.minlen:  # 最短长度判断
            best = output_scores.argmax()  # 得分最大的那个
            if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                break_tag = True
            elif not flag.all():  # 如果有已完成的
                inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                output_ids = output_ids[flag]  # 扔掉已完成序列
                output_scores = output_scores[flag]  # 扔掉已完成序列
                end_counts = end_counts[flag]  # 扔掉已完成end计数
                topk = flag.sum()  # topk相应变化
        return inputs, output_ids, output_scores, topk, results, break_tag

    def __batch_beam_search_step(self, step, inputs, output_ids, output_scores, topks, states, temperature):
        '''beam_search batch条推理计算得分'''
        self.step = step
        scores, states = self.predict(inputs, output_ids, states, temperature, 'logits')  # 计算当前得分
        if step == 0:  # 第0步预测后将输入重复topk次
            inputs_new = []
            for input_ in inputs:
                inputs_ = []
                for topk, input_i in zip(topks, input_):
                    input_i = input_i.unsqueeze(0)
                    inputs_.append(input_i.repeat([topk]+[1]*(len(input_i.shape)-1)))
                inputs_new.append(torch.cat(inputs_))
            inputs = inputs_new
            # 对seq_len进行扩充
            input_seqlen = []
            for topk, input_seqlen_i in zip(topks, self.input_seqlen):
                input_seqlen.append(input_seqlen_i.repeat(topk))
            self.input_seqlen = torch.cat(input_seqlen)

        scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
        output_ids_new, output_scores_new = [], []
        for smp_i, topk in enumerate(topks):
            if step == 0:
                score = scores[smp_i][None, ...]
                output_id = output_ids[smp_i][None, ...]
            else:
                start, end = sum(topks[:smp_i]), sum(topks[:smp_i+1])
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

    def __batch_beam_search_end(self, inputs, output_ids, output_scores, topks, results, min_ends):
        break_tag = False
        is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
        end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        self.flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列

        if output_ids.shape[1] >= self.minlen:  # 最短长度判断
            inputs_new, output_ids_new, output_scores_new, flag_new = [], [], [], []
            topks_new = topks.copy()
            for smp_i, topk in enumerate(topks):
                if topk == 0:
                    continue
                start, end = sum(topks[:smp_i]), sum(topks[:smp_i+1])
                input_ = [i[start:end] for i in inputs]
                output_score = output_scores[start:end]
                output_id = output_ids[start:end]

                best = output_score.argmax()  # 得分最大的那个
                is_end = output_id[:, -1] == self.end_id  # 标记是否以end标记结束
                end_counts = (output_id == self.end_id).sum(1)  # 统计出现的end标记
                flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
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
            topks = topks_new
            self.flag = torch.cat(flag_new)

            if len(output_ids) == 0:
                break_tag = True
        else:
            # 不满足结束条件，需要对self.flag进行更新
            self.flag = torch.ones_like(self.flag, dtype=torch.bool)
        
        return inputs, output_ids, output_scores, topks, results, break_tag

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        """beam search解码
        
        :param inputs: 编码器的输入，包含encoder_hidden_states, encoder_attention_mask
        :param topk: int, 这里的topk即beam size
        :param states:
        :param temperature: 温度参数，默认为1
        :param min_ends:
        :return: 最优解码序列。
        """
        assert topk is not None, 'Arg `topk` means beam_size anc can not be None'
        inputs = self._prepare_raw_inputs(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        output_scores = torch.zeros(btz, device=self.device)
        results = []

        # batch推理
        if self.use_batch:
            topk = [topk] * btz
            results = [None] * btz

        for step in range(self.maxlen):
            if (not self.use_batch):  # 单条推理
                inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, topk, states, temperature)
                inputs, output_ids, output_scores, topk, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, topk, results, min_ends)
            else:  # batch推理
                inputs, output_ids, output_scores, states = self.__batch_beam_search_step(step, inputs, output_ids, output_scores, topk, states, temperature)
                inputs, output_ids, output_scores, topk, results, break_tag = self.__batch_beam_search_end(inputs, output_ids, output_scores, topk, results, min_ends)
            if break_tag:
                break 
        # 如果还有未完成序列，直接放入结果
        if len(output_ids) > 0:
            if not self.use_batch:
                results.append(output_ids[output_scores.argmax()])
            elif self.use_batch:
                for smp_i, result in enumerate(results):
                    if result is None:
                        start, end = sum(topk[:smp_i]), sum(topk[:smp_i+1])
                        output_score = output_scores[start:end]
                        output_id = output_ids[start:end]
                        results[smp_i] = output_id[output_score.argmax()]

        # 达到长度直接输出
        self.flag = None
        return results

    def stream_beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        '''beam_search的stream输出模式'''
        assert topk is not None, 'Arg `topk` means beam_size anc can not be None'
        inputs = self._prepare_raw_inputs(inputs)
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        output_scores = torch.zeros(btz, device=self.device)
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, output_scores, states = self.__beam_search_step(step, inputs, output_ids, output_scores, topk, states, temperature)
            yield [output_ids[output_scores.argmax()]]
            inputs, output_ids, output_scores, topk, results, break_tag = self.__beam_search_end(inputs, output_ids, output_scores, topk, results, min_ends)
            if break_tag:
                break 
            
        # 达到长度直接输出
        self.flag = None

    def __random_sample_step(self, step, n, inputs, output_ids, states, temperature, topk, topp):
        '''为了random_sample和stream_random_sample共用，抽离出来的单步逻辑'''
        self.step = step
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
        return inputs, output_ids, states

    def __random_sample_end(self, inputs, output_ids, results, min_ends, smp_indexs=None):
        break_tag = False
        is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
        end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        f_flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
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

    def random_sample(self, inputs_raw, n, topk=50, topp=1, states=None, temperature=1, min_ends=1):
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
                inputs, output_ids, states = self.__random_sample_step(step, n, inputs, output_ids, states, temperature, topk, topp)
                inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results, min_ends)
            else:  # batch
                inputs, output_ids, states = self.__random_sample_step(step, n, inputs, output_ids, states, temperature, topk, topp)
                inputs, output_ids, results, break_tag, smp_indexs = self.__random_sample_end(inputs, output_ids, results, min_ends, smp_indexs)
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
    
    def stream_random_sample(self, inputs_raw, topk=50, topp=1, states=None, temperature=1, min_ends=1):
        """随机采样n个结果；stream输出"""
        inputs = self._prepare_raw_inputs(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, states = self.__random_sample_step(step, 1, inputs, output_ids, states, temperature, topk, topp)
            yield output_ids
            inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results, min_ends)
            if break_tag:
                break
        self.flag = None


class SeqGeneration(AutoRegressiveDecoder):
    '''单向decoder语言模型的解码，对AutoRegressiveDecoder的简单封装，可以使用cache来加快解码
    '''
    def __init__(self, model, tokenizer, start_id, end_id, maxlen, minlen=1, pad_id=None, pad_mode='post', mode='random_sample', default_rtype='logits', use_states=True):
        '''
        :param model: 模型
        :param tokenizer: tokenizer，如果使用第三方的tokenize，需要继承重写下pre_process和post_process
        :param start_id: int, decoder初始的token_id
        :param end_id: int, 结束解码的token_id
        :param pad_id: int, pad_id，在batch解码时候使用
        :param pad_mode: str, padding在前面还是后面，pre或者post
        :param maxlen: int, 最大解码长度
        :param minlen: int, 最小解码长度
        :param default_rtype: str, 模型输出的结果是logits设置为logits，如果是probas设置为probas
        :param use_states: str, 是否使用cache
        '''
        
        # 去除了device入参，因为可以直接使用传入的model.device
        pad_id = pad_id or tokenizer.pad_token_id
        super().__init__(start_id, end_id, maxlen, minlen, pad_id, pad_mode, next(model.parameters()).device)
        self.encoder = None
        self.decoder = model
        self.tokenizer = tokenizer
        assert mode in {'random_sample', 'beam_search'}, 'Args `mode` only support `random_sample/beam_search`.'
        self.mode = mode
        self.predict.set_default_rtype(default_rtype)  # 动态修改闭包中的default_rtype
        self.predict.set_use_states(use_states)  # 动态修改闭包中的use_states
        self.use_states = use_states
        self.use_segment_ids = hasattr(model, 'segment_vocab_size') and (model.segment_vocab_size > 0)  # 是否使用segment_ids
    
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

            # shape和size不一致: step=1时候，beam_search的btz=束宽
            if (self.mode == 'beam_search') and (self.step == 1):
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

    def pre_process(self, text):
        '''前处理，可以继承后自定义，主要用于第三方tokenizer的encode'''
        inputs = self.tokenizer.encode(text, maxlen=self.maxlen)
        return inputs if self.use_segment_ids else [inputs[0]]
    
    def post_process(self, output_ids):
        '''后处理，可以继承后自定义，主要用于第三方tokenizer的decode'''
        if len(output_ids) > 1:
            return [self.tokenizer.decode(ids.cpu().numpy()) for ids in output_ids]
        elif len(output_ids) == 1:
            return self.tokenizer.decode(output_ids[0].cpu().numpy())
        return output_ids

    def _generate(self, inputs, n, topk, topp, temperature, min_ends):
        if self.mode == 'random_sample':
            output_ids = self.random_sample(inputs, n, topk=topk, topp=topp, temperature=temperature, min_ends=min_ends)  # 基于随机采样
        elif self.mode == 'beam_search':
            output_ids = self.beam_search(inputs, topk=topk, temperature=temperature, min_ends=min_ends)  # 基于beam search
        return output_ids

    def generate(self, text, n=1, topk=50, topp=1, temperature=1, min_ends=1):
        '''单条样本生成'''
        assert isinstance(text, str), 'Arg `text` must be str format'
        self.use_batch = False
        inputs = self.pre_process(text)
        output_ids = self._generate(inputs, n, topk, topp, temperature, min_ends)
        return self.post_process(output_ids)

    def batch_generate(self, text_list, topk=50, topp=1, temperature=1, min_ends=1):
        '''batch样本生成，use_states=True时要求pad_mode='pre', use_states=False时候对'''
        # 参数设定
        assert isinstance(text_list, (list,tuple)), 'Arg `text_list` must be list/tuple format'
        self.use_batch = True
        if self.use_states and (self.pad_mode in {'post', 'right'}):
            self.pad_mode = 'pre'
            print("[WARNING] When arg `use_states`=True, you may set `pad_mode`='pre' to avoid error output, reset `pad_mode`='pre' instead")

        # 主流程
        inputs = self.pre_process(text_list)
        output_ids = self._generate(inputs, 1, topk, topp, temperature, min_ends)
        return self.post_process(output_ids)

    def stream_generate(self, text:str, topk=50, topp=1, temperature=1, min_ends=1):
        '''单条样本stream输出预测的结果'''
        assert isinstance(text, str), 'Arg `text` must be str format'
        self.use_batch = False
        inputs = self.pre_process(text)
        if self.mode == 'random_sample':
            for output_ids in  self.stream_random_sample(inputs, topk=topk, topp=topp, temperature=temperature, min_ends=min_ends):  # stream随机采样
                yield self.post_process(output_ids)
        elif self.mode == 'beam_search':
            for output_ids in  self.stream_beam_search(inputs, topk=topk, temperature=temperature, min_ends=min_ends):  # stream随机采样
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
            
    def generate(self, text, n=1, topk=50, topp=1, temperature=1, min_ends=1):
        assert isinstance(text, str), 'Arg `text` must be str format'
        self.use_batch = False
        inputs = self.pre_process(text)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        output_ids = super()._generate(encoder_output, n, topk, topp, temperature, min_ends)
        return self.post_process(output_ids)

    def batch_generate(self, text_list, topk=50, topp=1, temperature=1, min_ends=1):
        '''batch样本生成'''
        # 参数设定
        assert isinstance(text_list, (list,tuple)), 'Arg `text_list` must be list/tuple format'
        self.use_batch = True
        inputs = self.pre_process(text_list)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        output_ids = super()._generate(encoder_output, 1, topk, topp, temperature, min_ends)
        return self.post_process(output_ids)

    def stream_generate(self, text, topk=50, topp=1, temperature=1, min_ends=1):
        '''stream输出t预测的结果'''
        assert isinstance(text, str), 'Arg `text` must be str format'
        self.use_batch = False
        inputs = self.pre_process(text)
        inputs = self._prepare_raw_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        if self.mode == 'random_sample':
            for output_ids in  self.stream_random_sample(encoder_output, topk=topk, topp=topp, temperature=temperature, min_ends=min_ends):  # stream随机采样
                yield self.post_process(output_ids)
        elif self.mode == 'beam_search':
            for output_ids in  self.stream_beam_search(encoder_output, topk=topk, temperature=temperature, min_ends=min_ends):  # stream随机采样
                yield self.post_process(output_ids)
