'''自回归模型的生成
'''

import torch
import torch.nn as nn
import numpy as np
from bert4torch.snippets import take_along_dim, torch_div, sequence_padding

class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略

    :param start_id: int, 解码使用的起始token_id，不同预训练模型设置可能不一样
    :param end_id: int, 解码使用的结束token_id，不同预训练模型设置可能不一样
    :param maxlen: int, 最大解码长度
    :param minlen: int, 最小解码长度, 默认为1
    :param device: str, 默认为'cpu'
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

    def process_inputs(self, inputs_raw) -> list:
        '''对输入进行处理
        '''
        # 传入的Tensor直接[]后返回
        if isinstance(inputs_raw, torch.torch.Tensor):
            return [inputs_raw.to(self.device)]
        inputs = []
        for input_ in inputs_raw:
            if isinstance(input_, torch.torch.Tensor):
                pass
            elif isinstance(input_, (list, tuple, np.ndarray)):
                # 单条样本为[1,2,3]格式，需转为[[1,2,3]]
                input_ = input_ if all(isinstance(j, (tuple,list)) for j in input_) else [input_]
                input_ = torch.tensor(sequence_padding(input_, value=self.pad_id), device=self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(input_)
        return inputs

    @staticmethod
    def topk_topp_wraper(scores, topk=None, topp=None, min_tokens_to_keep=1):
        '''从hf移植来的简易版本, 暂未使用
        '''
        if topk is not None:
            indices_to_remove = scores < torch.topk(scores, topk)[0][..., -1, None]
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))

        if topp is not None:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - topp)
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep
                sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))
        return scores

    def beam_search(self, inputs_raw, topk=50, states=None, temperature=1, min_ends=1):
        """beam search解码
        
        :param inputs_raw: tensor、array、list、tuple, 解码的输入，一般为last_hidden_state, shape=[btz, seq_len, hdsz]
        :param topk: int, 这里的topk即beam size
        :param states:
        :param temperature: 温度参数，默认为1
        :param min_ends:
        :return: 最优解码序列。
        """
        inputs = self.process_inputs(inputs_raw)  # 对输入进行处理
        output_ids, output_scores = self.first_output_ids, torch.zeros(1, device=self.device)
        for step in range(self.maxlen):
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
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        self.flag = flag  # 记录已经完成序列
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        self.flag = None
        return output_ids[output_scores.argmax()]
    
    def batch_beam_search(self, inputs_raw, topk=50, states=None, temperature=1, min_ends=1):
        """beam search解码, batch版本
        """
        return

    def __random_sample_step(self, step, n, inputs, output_ids, states, temperature, topk, topp):
        '''为了random_sample和stream_random_sample共用，抽离出来的单步逻辑
        '''
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

    def __random_sample_end(self, inputs, output_ids, results, min_ends):
        break_tag = False
        is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
        end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        if output_ids.shape[1] >= self.minlen:  # 最短长度判断
            flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
            if flag.any():  # 如果有已完成的
                self.flag = flag  # 记录已经完成序列
                for ids in output_ids[flag]:  # 存好已完成序列
                    results.append(ids)
                flag = (flag == False)  # 标记未完成序列
                inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                output_ids = output_ids[flag]  # 只保留未完成部分候选集
                end_counts = end_counts[flag]  # 只保留未完成部分end计数
                if len(output_ids) == 0:
                    break_tag = True

        return inputs, output_ids, results, break_tag

    def random_sample(self, inputs_raw, n, topk=50, topp=1.0, states=None, temperature=1, min_ends=1):
        """随机采样n个结果；
        说明: 非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        
        :param inputs_raw: tensor、array、list、tuple, 解码的输入，一般为last_hidden_state, shape=[btz, seq_len, hdsz]
        :param topk: int, 这里的topk是指仅保留topk的值，hf中默认为50
        :param topp: float, 这里的topp是token的概率阈值设置，hf中默认值为1.0
        :param states:
        :param temperature: 温度参数，默认为1
        :param min_ends:
        :return: n个解码序列组成的list。
        """
        inputs = self.process_inputs(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        btz = inputs[0].shape[0]
        output_ids = self.first_output_ids.repeat(btz, 1)
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, states = self.__random_sample_step(step, n, inputs, output_ids, states, temperature, topk, topp)
            inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results, min_ends)
            if break_tag:
                break

        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        self.flag = None
        return results
    
    def stream_random_sample(self, inputs_raw, topk=50, topp=1.0, states=None, temperature=1, min_ends=1):
        """随机采样n个结果；stream输出
        """
        n = 1
        inputs = self.process_inputs(inputs_raw)  # 对输入进行处理
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            inputs, output_ids, states = self.__random_sample_step(step, n, inputs, output_ids, states, temperature, topk, topp)
            yield output_ids
            inputs, output_ids, results, break_tag = self.__random_sample_end(inputs, output_ids, results, min_ends)
            if break_tag:
                break
        self.flag = None


class SeqGeneration(AutoRegressiveDecoder):
    '''单向decoder语言模型的解码，对AutoRegressiveDecoder的简单封装，可以使用cache来加快解码
    '''
    def __init__(self, model, tokenizer, start_id, end_id, maxlen, minlen=1, pad_id=0, mode='random_sample', default_rtype='logits', use_states=True):
        '''
        :param model: 模型
        :param tokenizer: tokenizer，如果使用第三方的tokenize，需要继承重写下pre_process和post_process
        :param start_id: int, decoder初始的token_id
        :param end_id: int, 结束解码的token_id
        :param maxlen: int, 最大解码长度
        :param minlen: int, 最小解码长度
        :param default_rtype: str, 模型输出的结果是logits设置为logits，如果是probas设置为probas
        :param use_states: str, 是否使用cache
        '''
        
        # 去除了device入参，因为可以直接使用传入的model.device
        super().__init__(start_id, end_id, maxlen, minlen, next(model.parameters()).device)
        self.encoder = None
        self.decoder = model
        self.tokenizer = tokenizer
        assert mode in {'random_sample', 'beam_search'}, 'Args `mode` only support `random_sample/beam_search`.'
        self.mode = mode
        self.pad_id = pad_id
        self.predict.set_default_rtype(default_rtype)  # 动态修改闭包中的default_rtype
        self.predict.set_use_states(use_states)  # 动态修改闭包中的use_states
        self.use_states = use_states
        self.use_segment_ids = hasattr(model, 'segment_vocab_size') and (model.segment_vocab_size > 0)  # 是否使用segment_ids
    
    def concat_token_ids(self, token_ids, output_ids):
        '''把非padding部分concat在一起
        '''
        inputs = []
        for token_ids_i, output_ids_i in zip(token_ids, output_ids):
            pad_index = (token_ids_i != self.pad_id).sum()
            inputs.append(torch.cat([token_ids_i[:pad_index], output_ids_i, token_ids_i[pad_index:]]))
        return torch.stack(inputs)

    def prepare_inputs(self, inputs, output_ids, include_past=True):
        if include_past is False:
            if len(inputs) == 1:
                inputs = [output_ids[:, -1:]]
            elif len(inputs) >= 2:  # 第二个是segment_ids
                token_ids = inputs[0]
                curr_segment_ids = token_ids[0, -1:]
                inputs = [output_ids[:, -1:], curr_segment_ids]
        else:
            if len(inputs) == 1:
                # token_ids = torch.cat([inputs[0], output_ids], 1)
                token_ids = self.concat_token_ids(inputs[0], output_ids)
                inputs = [token_ids]
            elif len(inputs) >= 2:  # 第二个是segment_ids
                token_ids, segment_ids = inputs
                # token_ids = torch.cat([token_ids, output_ids], 1)
                token_ids = self.concat_token_ids(inputs[0], output_ids)
                curr_segment_ids = torch.zeros_like(output_ids) + token_ids[0, -1]
                segment_ids = torch.cat([segment_ids, curr_segment_ids], 1)
                inputs = [token_ids, segment_ids]
        return inputs

    @AutoRegressiveDecoder.wraps()
    def predict(self, inputs, output_ids, states):
        assert isinstance(inputs, (tuple, list))
        if states is not None:
            assert self.use_states is True, 'Args `use_states` must be True when return states is not None'
        
        # 使用cache
        if self.use_states:
            token_ids = self.prepare_inputs(inputs, output_ids)[0]
            inputs = self.prepare_inputs(inputs, output_ids, include_past=(states is None))
            states = {'return_model_kwargs': True} if states is None else states
            # token_ids也返回下
            if self.step >= 1:
                states['token_ids'] = token_ids
            # attention_mask是根据token_ids生成的，因此这里重置下
            if states.get('input_attention_mask') is not None:  # 在states中input_attention_mask才是[btz, seq_len]
                attention_mask = states['input_attention_mask']
                states['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            # position_ids也需要修改下
            if states.get('position_ids') is not None:
                states['position_ids'] = states['position_ids'][:, -1:]+1
            # past_key_values和btz维度对不上
            if (states.get('past_key_values') is not None) and states['past_key_values'][0][0].shape[0] != output_ids.shape[0]:
                btz = output_ids.shape[0]
                if self.step == 1:  # beam_search在step=1时候btz=束宽
                    for l_i, past_key_value in enumerate(states['past_key_values']):
                        states['past_key_values'][l_i] = (past_key_value[0].repeat(btz, 1, 1, 1),
                                                        past_key_value[1].repeat(btz, 1, 1, 1))
                elif hasattr(self, 'flag'):  # 有的部分序列已经完成
                    for l_i, past_key_value in enumerate(states['past_key_values']):
                        states['past_key_values'][l_i] = (past_key_value[0][self.flag], past_key_value[1][self.flag])
                else:
                    raise ValueError("decoder output_ids and past_key_values's batch_size not same")
            logits, states = self.decoder.predict(inputs, **states)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return logits[:, -1, :], states

        # 不使用cache
        elif not self.use_states:
            inputs = self.prepare_inputs(inputs, output_ids, include_past=True)
            logits = self.decoder.predict(inputs)
            logits = logits[-1] if isinstance(logits, (tuple,list)) else logits  # 兼顾seq2seq
            return logits[:, -1, :]

    def pre_process(self, text):
        # 前处理，可以按照自定义
        inputs = self.tokenizer.encode(text, maxlen=self.maxlen)
        return inputs if self.use_segment_ids else [inputs[0]]
    
    def post_process(self, output_ids):
        # 后处理，可以按照自定义
        if len(output_ids) > 1:
            return [self.tokenizer.decode(ids.cpu().numpy()) for ids in output_ids]
        elif len(output_ids) == 1:
            return self.tokenizer.decode(output_ids[0].cpu().numpy())
        return output_ids

    def _generate(self, inputs, n, topk, topp, temperature):
        if self.mode == 'random_sample':
            output_ids = self.random_sample(inputs, n, topk=topk, topp=topp, temperature=temperature)  # 基于随机采样
        elif self.mode == 'beam_search':
            output_ids = [self.beam_search(inputs, topk=topk)]  # 基于beam search
        return output_ids

    def generate(self, text, n=1, topk=None, topp=None, temperature=1):
        '''样本生成
        '''
        if isinstance(text, (tuple,list)):  # batch预测n要设置为1
            n = 1
        inputs = self.pre_process(text)
        output_ids = self._generate(inputs, n, topk, topp, temperature)
        return self.post_process(output_ids)

    def stream_generate(self, text, topk=None, topp=None, temperature=1):
        '''单条样本stream输出预测的结果
        '''
        inputs = self.pre_process(text)
        for output_ids in  self.stream_random_sample(inputs, topk=topk, topp=topp, temperature=temperature):  # stream随机采样
            yield self.post_process(output_ids)

class Seq2SeqGeneration(SeqGeneration):
    '''encoder-decoder语言模型的解码
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.decoder.encoder
        self.decoder = self.decoder.decoder
        self.use_segment_ids = hasattr(self.encoder, 'segment_vocab_size') and (self.encoder.segment_vocab_size > 0)  # 是否使用segment_ids

    @staticmethod
    def prepare_inputs(encoder_outputs, decoder_inputs, include_past=True):
        if include_past is False:
            # 这里未做判断，因为一般seq2seq模型都没有segment_ids
            inputs = [decoder_inputs[:, -1:]] + encoder_outputs
        else:
            inputs = [decoder_inputs] + encoder_outputs
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        return inputs
            
    def generate(self, text, n=1, topk=None, topp=None, temperature=1):
        if isinstance(text, (tuple,list)):  # batch预测n要设置为1
            n = 1
        inputs = self.pre_process(text)
        inputs = self.process_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        output_ids = super()._generate(encoder_output, n, topk, topp, temperature)
        return self.post_process(output_ids)

    def stream_generate(self, text, topk=None, topp=None, temperature=1):
        '''stream输出t预测的结果
        '''
        inputs = self.pre_process(text)
        inputs = self.process_inputs(inputs)  # 有时候需要list转tensor
        encoder_output = self.encoder.predict(inputs)
        for output_ids in  self.stream_random_sample(encoder_output, topk=topk, topp=topp, temperature=temperature):  # stream随机采样
            yield self.post_process(output_ids)
