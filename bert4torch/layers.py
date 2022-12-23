import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from bert4torch.snippets import get_sinusoid_encoding_table, take_along_dim, torch_div
from bert4torch.activations import get_activation
from typing import List, Optional, Union
import random
import warnings


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, conditional_size=False, weight=True, bias=True, norm_mode='normal', **kwargs):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        
        # 兼容roformer_v2不包含weight
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        # 兼容t5不包含bias项, 和t5使用的RMSnorm
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm_mode = norm_mode

        self.eps = eps
        self.conditional_size = conditional_size
        if conditional_size:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, x):
        inputs = x[0]

        if self.norm_mode == 'rmsnorm':
            # t5使用的是RMSnorm
            variance = inputs.to(torch.float32).pow(2).mean(-1, keepdim=True)
            o = inputs * torch.rsqrt(variance + self.eps)
        else:
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            o = (inputs - u) / torch.sqrt(s + self.eps)

        if not hasattr(self, 'weight'):
            self.weight = 1
        if not hasattr(self, 'bias'):
            self.bias = 0

        if self.conditional_size:
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            return (self.weight + self.dense1(cond)) * o + (self.bias + self.dense2(cond))
        else:
            return self.weight * o + self.bias


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate=0.1, attention_scale=True,
                 return_attention_scores=False, bias=True, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # assert hidden_size % num_attention_heads == 0  # 旧逻辑，t5_pegasus_small中不可以整除
        # 兼容t5_pegasus_small
        if kwargs.get('attention_head_size'):
            self.attention_head_size = kwargs.get('attention_head_size')
        else:
            self.attention_head_size = int(hidden_size / num_attention_heads)
        self.inner_dim = self.num_attention_heads * self.attention_head_size  # 新逻辑
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.bias = bias
        self.q = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.k = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.v = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.o = nn.Linear(self.inner_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')

        if self.p_bias == 'typical_relative':  # nezha
            self.relative_positions_encoding = RelativePositionsEncoding(qlen=kwargs.get('max_position'),
                                                                         klen=kwargs.get('max_position'),
                                                                         embedding_size=self.attention_head_size,
                                                                         max_relative_position=kwargs.get('max_relative_position'))
        elif self.p_bias == 'rotary':  # roformer
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'), embedding_size=self.attention_head_size)
        elif self.p_bias == 't5_relative':  # t5
            self.relative_positions = RelativePositionsEncodingT5(qlen=kwargs.get('max_position'), 
                                                                  klen=kwargs.get('max_position'), 
                                                                  relative_attention_num_buckets=kwargs.get('relative_attention_num_buckets'), 
                                                                  is_decoder=kwargs.get('is_decoder'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'), self.num_attention_heads)

        elif self.p_bias == 'deberta_v2':  # deberta_v2
            # 配置文件
            self.share_att_key = kwargs.get("share_att_key", False)
            self.position_buckets = kwargs.get("position_buckets", -1)
            self.max_relative_positions = kwargs.get("max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = kwargs.get('max_position_embeddings')
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            # position_embedding
            self.pos_att_type = kwargs.get('pos_att_type', [])
            self.relative_positions = RelativePositionsEncodingDebertaV2(qlen=kwargs.get('max_position'), 
                                                                         klen=kwargs.get('max_position'), 
                                                                         position_buckets=kwargs.get('position_buckets'),
                                                                         max_position=kwargs.get('max_position'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('max_position'), self.hidden_size)
            self.norm_rel_ebd = [x.strip() for x in kwargs.get("norm_rel_ebd", "none").lower().split("|")]
            if "layer_norm" in self.norm_rel_ebd:
                self.layernorm = nn.LayerNorm(self.hidden_size, kwargs.get('layer_norm_eps', 1e-12), elementwise_affine=True)

            self.pos_dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, query_states=None):
        # hidden_states shape: [batch_size, seq_q, hidden_size]
        # attention_mask shape: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        # encoder_hidden_states shape: [batch_size, seq_k, hidden_size]
        # encoder_attention_mask shape: [batch_size, 1, 1, seq_k]

        if query_states is None:
            query_states = hidden_states  # 在deberta_v2中使用
        mixed_query_layer = self.q(query_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.k(encoder_hidden_states)
            mixed_value_layer = self.v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k(hidden_states)
            mixed_value_layer = self.v(hidden_states)
        # mixed_query_layer shape: [batch_size, query_len, hidden_size]
        # mixed_query_layer shape: [batch_size, key_len, hidden_size]
        # mixed_query_layer shape: [batch_size, value_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        if self.p_bias == 'rotary':
            query_layer = self.relative_positions_encoding(query_layer)
            key_layer = self.relative_positions_encoding(key_layer)

        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])  # [to_seq_len, to_seq_len, d_hid]
            # 旧实现，方便读者理解维度转换
            # query_layer_t = query_layer.permute(2, 0, 1, 3)
            # query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, self.attention_head_size)
            # key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
            # key_position_scores_r = key_position_scores.view(from_seq_length, batch_size, num_attention_heads, from_seq_length)
            # key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_layer, relations_keys)
            attention_scores = attention_scores + key_position_scores_r_t
        elif (self.p_bias == 't5_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
            attention_scores = attention_scores + key_position_scores_r_t

        # 是否进行attention scale
        if self.p_bias == 'deberta_v2':
            scale_factor = 1
            if "c2p" in self.pos_att_type:
                scale_factor += 1
            if "p2c" in self.pos_att_type:
                scale_factor += 1
            scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
            attention_scores = attention_scores / scale.to(dtype=query_layer.dtype)

            rel_embeddings = self.pos_dropout(self.layernorm(self.relative_positions_encoding.weight))
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relations_keys, rel_embeddings, scale_factor)
            attention_scores = attention_scores + rel_att

        elif self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
        if attention_mask is not None:
            # attention_mask = attention_mask * attention_mask.squeeze(-2).unsqueeze(-1)  # deberta_v2中使用，但是不使用也不影响
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)  # 下一行的另一种写法
            attention_mask = (1.0 - attention_mask) * -10000.0  # 所以传入的mask的非padding部分为1, padding部分为0
            attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_attention_heads, query_len, attention_head_size]

        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_values = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
            # 旧实现，方便读者理解维度转换
            # attention_probs_t = attention_probs.permute(2, 0, 1, 3)
            # attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, to_seq_length)
            # value_position_scores = torch.matmul(attentions_probs_r, relations_values)
            # value_position_scores_r = value_position_scores.view(from_seq_length, batch_size, num_attention_heads, self.attention_head_size)
            # value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', attention_probs, relations_values)
            context_layer = context_layer + value_position_scores_r_t

        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]
        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 所以在调用view之前，需要contiguous来返回一个contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.inner_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        '''deberta_v2使用，和原版区别是query_layer是4维
        '''
        btz, n_head, q_len, d_head = query_layer.size()
        k_len = key_layer.size(-2)
        if relative_pos is None:
            relative_pos = self.relative_positions(q_len, k_len)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.q(rel_embeddings)).repeat(btz, 1, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.k(rel_embeddings)).repeat(btz, 1, 1, 1)
        else:
            # 这里逻辑去掉了
            pass

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.expand([btz, n_head, q_len, k_len]))
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            if k_len != q_len:
                r_pos = self.relative_positions(k_len, k_len)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([btz, n_head, k_len, k_len])).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.5, hidden_act='gelu', is_dropout=False, bias=True, **kwargs):
        # 原生的tf版本的bert在激活函数后，没有添加dropout层，但是在google AI的bert-pytorch开源项目中，多了一层dropout；
        # 并且在pytorch官方的TransformerEncoderLayer的实现中，也有一层dropout层，就像这样：self.linear2(self.dropout(self.activation(self.linear1(src))))；
        # 这样不统一做法的原因不得而知，不过有没有这一层，差别可能不会很大；

        # 为了适配是否dropout，用is_dropout，dropout_rate两个参数控制；如果是实现原始的transformer，直接使用默认参数即可；如果是实现bert，则is_dropout为False，此时的dropout_rate参数并不会使用.
        super(PositionWiseFeedForward, self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = get_activation(hidden_act)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch size, seq len, hidden_size)
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))

        # x shape: (batch size, seq len, intermediate_size)
        x = self.outputDense(x)

        # x shape: (batch size, seq len, hidden_size)
        return x


class GatedAttentionUnit(nn.Module):
    '''门控注意力单元，
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码
    参考pytorch项目：https://github.com/lucidrains/FLASH-pytorch
    '''
    
    def __init__(self, hidden_size, attention_key_size, intermediate_size, attention_probs_dropout_prob, hidden_act, 
                 is_dropout=False, attention_scale=True, bias=True, normalization='softmax_plus', **kwargs):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.attention_head_size = attention_key_size
        self.attention_scale = attention_scale
        self.is_dropout = is_dropout
        self.normalization = normalization
        self.hidden_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.i_dense = nn.Linear(hidden_size, self.intermediate_size*2+attention_key_size, bias=bias)
        self.offsetscale = self.OffsetScale(attention_key_size, heads=2, bias=bias)
        self.o_dense = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        
        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if self.p_bias == 'rotary':  # RoPE
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'), embedding_size=self.attention_head_size)

    def forward(self, hidden_states, attention_mask):
        # 投影变换
        hidden_states = self.hidden_fn(self.i_dense(hidden_states))
        u, v, qk = hidden_states.split([self.intermediate_size, self.intermediate_size, self.attention_head_size], dim=-1)
        q, k = self.offsetscale(qk)  # 仿射变换

        # 加入RoPE
        if self.p_bias == 'rotary':
            q = self.relative_positions_encoding(q)
            k = self.relative_positions_encoding(k)

        # Attention
        attention_scores = torch.einsum('b i d, b j d -> b i j', q, k)  # [btz, seq_len, seq_len]
        if self.attention_scale:
            # seq_len = hidden_states.shape[1]
            # attention_scores = F.relu(attention_scores/seq_len) ** 2
             attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -1e12
            attention_scores = attention_scores + attention_mask.squeeze(1)

        # 归一化
        attention_scores = self.attention_normalize(attention_scores, -1, self.normalization)

        if self.is_dropout:
            attention_scores = self.dropout(attention_scores)

        # 计算输出
        out = self.o_dense(u * torch.einsum('b i j, b j d -> b i d', attention_scores, v))
        return out
    
    def attention_normalize(self, a, dim=-1, method='softmax'):
        """不同的注意力归一化方案
        softmax：常规/标准的指数归一化；
        squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
        softmax_plus：来自 https://kexue.fm/archives/8823 。
        """
        if method == 'softmax':
            return F.softmax(a, dim=dim)
        else:
            mask = (a > -1e11).float()
            l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1).to(mask))
            if method == 'squared_relu':
                return F.relu(a)**2 / l
            elif method == 'softmax_plus':
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512.0)).to(mask), dim=dim)
        return a

    class OffsetScale(nn.Module):
        '''仿射变换
        '''
        def __init__(self, head_size, heads=1, bias=True):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(heads, head_size))
            self.bias = bias
            if bias:
                self.beta = nn.Parameter(torch.zeros(heads, head_size))
            nn.init.normal_(self.gamma, std = 0.02)

        def forward(self, x):
            out = torch.einsum('... d, h d -> ... h d', x, self.gamma)
            if self.bias:
                 out = out + self.beta
            return out.unbind(dim = -2)


class BertEmbeddings(nn.Module):
    """embeddings层
       构造word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, max_position, segment_vocab_size, shared_segment_embeddings, dropout_rate, conditional_size=False, token_pad_ids=0, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=token_pad_ids)

        # 位置编码
        if kwargs.get('p_bias') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)
        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'other_relative', 'deberta_v2'}:
            # 如果使用相对位置编码，则不声明PositionEmbeddings
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        
        # segement编码
        if (segment_vocab_size > 0) and (not shared_segment_embeddings) and kwargs.get('use_segment_embedding', True):
            # use_segment_embedding用于lm, unilm场景，不使用segment_embeddings但是传入segment_ids用于计算mask
            # 一般无需设置，目前仅在guwenbert中使用
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        # emb_scale
        self.emb_scale = kwargs.get('emb_scale', 1)  # transform_xl, xlnet特有

        # LayerNorm
        self.layerNorm = LayerNorm(embedding_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

        # 如果embedding_size != hidden_size，则再有一个linear(适用于albert矩阵分解)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def forward(self, token_ids, segment_ids=None, position_ids=None, conditional_emb=None, additional_embs=None, attention_mask=None):
        if (not token_ids.requires_grad) and (token_ids.dtype in {torch.long, torch.int}):
            words_embeddings = self.word_embeddings(token_ids)
        else:
            words_embeddings = token_ids  # 自定义word_embedding，目前仅有VAT中使用

        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)  
            embeddings = words_embeddings + segment_embeddings
        elif self.shared_segment_embeddings:  # segment和word_embedding共享权重
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)  
            embeddings = words_embeddings + segment_embeddings
        else:
            embeddings = words_embeddings
        
        # 额外的embedding，如词性等
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb

        if hasattr(self, 'position_embeddings'):
            seq_length = token_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
                position_ids = position_ids.unsqueeze(0).repeat(token_ids.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale  # transform_xl, xlnet特有

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm((embeddings, conditional_emb))
        
        if attention_mask is not None:
            embeddings *= attention_mask[:, 0, 0, :, None]

        embeddings = self.dropout(embeddings)

        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """Transformer层:
       顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

       注意:
        1. 以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
        2. 原始的Transformer的encoder中的Feed Forward层一共有两层linear，
        3. config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, 
                 is_dropout=False, conditional_size=False, **kwargs):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act, is_dropout=is_dropout, **kwargs)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.is_decoder = kwargs.get('is_decoder')
        if self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.layerNorm3 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)

    def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)  # self.decoder为true时候，这里的attention_mask是三角的
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1((hidden_states, conditional_emb))
        
        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output = self.crossAttention(hidden_states, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)
            hidden_states = self.layerNorm3((hidden_states, conditional_emb))
            
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states


class T5Layer(BertLayer):
    """T5的Encoder的主体是基于Self-Attention的模块
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """
    def __init__(self, *args, version='t5.1.0', **kwargs):
        super().__init__(*args, **kwargs)

        # 如果是t5.1.1结构，则FFN层需要变更
        if version.endswith('t5.1.1'):
            kwargs['dropout_rate'] = args[2]
            kwargs['hidden_act'] = args[5]
            self.feedForward = self.T5PositionWiseFeedForward(hidden_size=args[0], intermediate_size=args[4], **kwargs)

        # decoder中间有crossAttention
        if self.is_decoder and hasattr(self.crossAttention, 'relative_positions_encoding'):
            del self.crossAttention.relative_positions_encoding
            del self.crossAttention.relative_positions

    def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
        x = self.layerNorm1((hidden_states, conditional_emb))
        self_attn_output = self.multiHeadAttention(x, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)

        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.layerNorm3((hidden_states, conditional_emb))
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)

        x = self.layerNorm2((hidden_states, conditional_emb))
        ffn_output = self.feedForward(x)
        hidden_states = hidden_states + self.dropout2(ffn_output)
        return hidden_states

    class T5PositionWiseFeedForward(PositionWiseFeedForward):
        '''参考transformer包: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
        '''
        def __init__(self, hidden_size, intermediate_size, **kwargs):
            super().__init__(hidden_size, intermediate_size, **kwargs)
            self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.intermediateDense1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            # x shape: (batch size, seq len, hidden_size)
            x_gelu = self.intermediate_act_fn(self.intermediateDense(x))
            x_linear = self.intermediateDense1(x)
            x = x_gelu * x_linear
            if self.is_dropout:
                x = self.dropout(x)

            # x shape: (batch size, seq len, intermediate_size)
            x = self.outputDense(x)

            # x shape: (batch size, seq len, hidden_size)
            return x


class XlnetLayer(BertLayer):
    '''Transformer_XL层
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    '''
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs):
        super().__init__(hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs)
        self.pre_lnorm = kwargs.get('pre_lnorm')
        # multiattn层无bias
        self.multiHeadAttention = self.RelPartialLearnableMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, bias=False, **kwargs)

    def forward(self, hidden_states, segment_ids, pos_emb, attention_mask, mems_i, conditional_emb=None):
        # 拼接mems和query，mems_i: [btz, m_len, hdsz], w: [btz, q_len, hdsz] = [btz, k_len, hdsz]
        hidden_states_cat = torch.cat([mems_i, hidden_states], 1) if mems_i is not None else hidden_states
        
        # Attn
        if self.pre_lnorm:
            hidden_states_cat = self.layerNorm1((hidden_states_cat, conditional_emb))
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states_cat, pos_emb, attention_mask, segment_ids)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        if not self.pre_lnorm:  # post_lnorm
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))

        # FFN
        x = self.layerNorm2((hidden_states, conditional_emb)) if self.pre_lnorm else hidden_states
        self_attn_output2 = self.feedForward(x)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        if not self.pre_lnorm:  # post_lnorm
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states

    class RelPartialLearnableMultiHeadAttn(MultiHeadAttentionLayer):
        '''Transformer_XL式相对位置编码, 这里修改成了MultiHeadAttentionLayer的batch_first代码格式
        '''
        def __init__(self, *args, r_w_bias=None, r_r_bias=None, r_s_bias=None, **kwargs):
            super().__init__(*args, **kwargs)
            segment_vocab_size = kwargs.get('segment_vocab_size')
            if r_r_bias is None or r_w_bias is None:  # Biases are not shared
                self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局内容偏置
                self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局位置偏置
                if segment_vocab_size > 0:
                    self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局segment偏置
            else:  # 所有层公用一个
                self.r_r_bias = r_r_bias
                self.r_w_bias = r_w_bias
                self.r_s_bias = r_s_bias
            if segment_vocab_size > 0:
                # self.seg_embed = nn.Embedding(segment_vocab_size, self.hidden_size)
                self.seg_embed = nn.Parameter(torch.FloatTensor(segment_vocab_size, self.num_attention_heads, self.attention_head_size))

            self.r = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.rel_shift_opt = kwargs.get('rel_shift_opt')

        @staticmethod
        def rel_shift(x, zero_triu=False):
            '''transformer_xl使用, 向左shift让右上角都是0, 对角线是同一个值, x: [btz, n_head, q_len, k_len]
            '''
            q_len, k_len = x.size(2), x.size(-1)
            zero_pad = torch.zeros((*x.size()[:2], q_len, 1), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=-1)
            x_padded = x_padded.view(*x.size()[:2], k_len + 1, q_len)
            x = x_padded[:,:,1:,:].view_as(x)
            if zero_triu:
                ones = torch.ones((q_len, k_len), device=x.device)
                x = x * torch.tril(ones, k_len - q_len)[None,None,:,:]
            return x

        @staticmethod
        def rel_shift_bnij(x, klen=-1):
            ''' xlnet使用
            '''
            x_size = x.shape
            x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
            x = x[:, :, 1:, :]
            x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
            x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
            # x = x[:, :, :, :klen]
            return x

        def forward(self, w, cat, r, attention_mask=None, seg_mat=None):
            # w: 词向量[btz, q_len, hdsz], cat: w和mem_i拼接后向量[btz, k_len, hdsz], r：相对位置向量[r_len, hdsz]
            qlen, rlen, bsz = w.size(1), r.size(0), w.size(0)
            
            mixed_query_layer = self.q(cat)[:, -qlen:, :]  # 仅取用query部分，不适用mem部分
            mixed_key_layer = self.k(cat)
            mixed_value_layer = self.v(cat)

            w_head_q = self.transpose_for_scores(mixed_query_layer)  # [btz, n_head, q_len, d_head]
            w_head_k = self.transpose_for_scores(mixed_key_layer)  # [btz, n_head, k_len, d_head]
            w_head_v = self.transpose_for_scores(mixed_value_layer)  # [btz, n_head, k_len, d_head]

            r_head_k = self.r(r)  # [hdsz, nhead*headsize] = [r_len, 1, nhead*headsize]
            r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)  # rlen x n_head x d_head

            #### compute attention score
            rw_head_q = w_head_q + self.r_w_bias.unsqueeze(1)  # [btz, n_head, q_len, d_head]
            AC = torch.einsum('bnid,bnjd->bnij', (rw_head_q, w_head_k))  # [btz, n_head, q_len, k_len]

            rr_head_q = w_head_q + self.r_r_bias.unsqueeze(1)  # [btz, n_head, q_len, d_head]
            BD = torch.einsum('bnid,jnd->bnij', (rr_head_q, r_head_k))  # [btz, n_head, q_len, k_len]
            BD = self.rel_shift_bnij(BD, klen=AC.shape[3]) if self.rel_shift_opt == 'xlnet' else self.rel_shift(BD)

            if hasattr(self, 'seg_embed') and (self.r_r_bias is not None):
                # # 之前的方式，需要配合Embedding，以及load_variable和variable_mapping，显存容易爆炸
                # w_head_s = self.seg_embed(seg_mat)  # [btz, q_len, klen, hdsz]
                # w_head_s = w_head_s.reshape(*w_head_s.shape[:3], self.num_attention_heads, self.attention_head_size)
                # rs_head_q = w_head_q + self.r_s_bias.unsqueeze(1)
                # EF = torch.einsum('bnid,bijnd->bnij', (rs_head_q, w_head_s))  # [btz, n_head, q_len, k_len]
                
                seg_mat = F.one_hot(seg_mat, 2).float()
                EF = torch.einsum("bnid,snd->ibns", w_head_q + self.r_s_bias.unsqueeze(1), self.seg_embed)
                EF = torch.einsum("bijs,ibns->bnij", seg_mat, EF)
            else:
                EF = 0

            # # [btz, n_head, q_len, k_len]
            attention_scores = AC + BD + EF
            if self.attention_scale:
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            #### compute attention probability
            if attention_mask is not None and attention_mask.any().item():
                # attention_mask = (1.0 - attention_mask) * -10000.0
                # attention_scores = attention_scores + attention_mask  # 这里修改了下，原有的-10000不够接近-inf
                attention_mask = (1.0 - attention_mask)
                attention_scores = attention_scores.float().masked_fill(attention_mask.bool(), -1e30).type_as(attention_mask)

            # [btz, n_head, q_len, k_len]
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, w_head_v)  # [batch_size, num_attention_heads, query_len, attention_head_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # 是否返回attention scores
            if self.return_attention_scores:
                # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
                return self.o(context_layer), attention_scores
            else:
                return self.o(context_layer)


class AdaptiveEmbedding(nn.Module):
    '''Transformer_XL的自适应embedding, 实现不同区间使用不同的维度
    可以实现如高频词用比如1024或512维，低频词用256或64维, 再用Linear层project到相同的维数
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, cutoffs, div_val=1, sample_softmax=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, embedding_size, sparse=sample_softmax > 0))
            if hidden_size != embedding_size:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, embedding_size)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = embedding_size // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, d_emb_i)))

    def forward(self, token_ids):
        if self.div_val == 1:  # 仅有一个embedding
            embed = self.emb_layers[0](token_ids)  # [btz, seq_len, embedding_size]
            if self.hidden_size != self.embedding_size:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = token_ids.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.hidden_size], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = token_ids.size() + (self.hidden_size,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)

        return embed


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]


class XlnetPositionsEncoding(nn.Module):
    '''Xlnet, transformer_xl使用的相对位置编码
       和SinusoidalPositionEncoding区别是一个是间隔排列, 一个是前后排列
    '''
    def __init__(self, embedding_size):
        super().__init__()
        self.demb = embedding_size
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_size, 2.0) / embedding_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class RelativePositionsEncodingDebertaV2(nn.Module):
    """deberta用的相对位置编码
    来自论文：https://arxiv.org/abs/2006.03654
    """
    def __init__(self, qlen, klen, position_buckets, max_position):
        super(RelativePositionsEncodingDebertaV2, self).__init__()
        q_ids = torch.arange(0, qlen)
        k_ids = torch.arange(0, klen)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        if position_buckets > 0 and max_position > 0:
            rel_pos_ids = self.make_log_bucket_position(rel_pos_ids, position_buckets, max_position)
        rel_pos_ids = rel_pos_ids.to(torch.long)
        rel_pos_ids = rel_pos_ids[:qlen, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        self.register_buffer('relative_position', rel_pos_ids)

    @staticmethod
    def make_log_bucket_position(relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid),
            torch.tensor(mid - 1).type_as(relative_pos),
            torch.abs(relative_pos),
        )
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
        return bucket_pos

    def forward(self, qlen, klen):
        return self.relative_position[:, :qlen, :klen]


class RelativePositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(RelativePositionsEncoding, self).__init__()
        # 生成相对位置矩阵
        vocab_size = max_relative_position * 2 + 1
        distance_mat = torch.arange(klen)[None, :] - torch.arange(qlen)[:, None]  # 列数-行数, [query_len, key_len]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        # sinusoid_encoding编码的位置矩阵
        embeddings_table = get_sinusoid_encoding_table(vocab_size, embedding_size)

        # 实现方式1
        # flat_relative_positions_matrix = final_mat.view(-1)
        # one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix, num_classes=vocab_size).float()
        # position_embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        # my_shape = list(final_mat.size())
        # my_shape.append(embedding_size)
        # position_embeddings = position_embeddings.view(my_shape)

        # 实现方式2
        # position_embeddings = take_along_dim(embeddings_table, final_mat.flatten().unsqueeze(1), dim=0)
        # position_embeddings = position_embeddings.reshape(*final_mat.shape, embeddings_table.shape[-1])  # [seq_len, seq_len, hdsz]
        # self.register_buffer('position_embeddings', position_embeddings)
        
        # 实现方式3
        position_embeddings = nn.Embedding.from_pretrained(embeddings_table, freeze=True)(final_mat)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, qlen, klen):
        return self.position_embeddings[:qlen, :klen, :]


class RelativePositionsEncodingT5(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(RelativePositionsEncodingT5, self).__init__()
        # 生成相对位置矩阵
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        relative_position = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not is_decoder,
            num_buckets=relative_attention_num_buckets,
        )
        self.register_buffer('relative_position', relative_position)

    def forward(self, qlen, klen):
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        '''直接来源于transformer
        '''
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=True) 
    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)
    
    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class CRF(nn.Module):
    '''Conditional random field: https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/layers/crf.py
    '''
    def __init__(self, num_tags: int, init_transitions: Optional[List[np.ndarray]] = None, freeze=False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        if (init_transitions is None) and (not freeze):
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
        elif init_transitions is not None:
            transitions = torch.tensor(init_transitions[0], dtype=torch.float)
            start_transitions = torch.tensor(init_transitions[1], dtype=torch.float)
            end_transitions = torch.tensor(init_transitions[2], dtype=torch.float)

            if not freeze:
                self.transitions = nn.Parameter(transitions)
                self.start_transitions = nn.Parameter(start_transitions)
                self.end_transitions = nn.Parameter(end_transitions)
            else:
                self.register_buffer('transitions', transitions)
                self.register_buffer('start_transitions', start_transitions)
                self.register_buffer('end_transitions', end_transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, mask: torch.ByteTensor,
                tags: torch.LongTensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            emissions: [btz, seq_len, num_tags]
            mask: [btz, seq_len]
            tags: [btz, seq_len]
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        best_path = self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)
        return best_path[0] if nbest == 1 else best_path

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, '
                             f'got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError('the first two dimensions of emissions and tags must match, '
                                 f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError('the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq_bf = mask[:, 0].all()
            if not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)
        # tags: (batch_size, seq_length)
        # mask: (batch_size, seq_length)
        batch_size, seq_length = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1
        # shape: (batch_size,)
        last_tags = tags[torch.arange(batch_size), seq_ends]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)
        # mask: (batch_size, seq_length)
        seq_length = emissions.size(1)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
                              nbest: int, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (batch_size, seq_length, num_tags)
        # mask: (batch_size, seq_length)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        batch_size, seq_length = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[:, 0]
        history_idx = torch.zeros((batch_size, seq_length, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((batch_size, seq_length, nbest), pad_tag, dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[:, i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((batch_size, seq_length, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[:, idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[:, idx] = torch_div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')  # 兼容老版本
            # best_tags_arr[:, idx] = torch.div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')

        return torch.where(mask.unsqueeze(-1).bool(), best_tags_arr, oor_tag).permute(2, 0, 1)


class BERT_WHITENING():
    def __init__(self):
        self.kernel = None
        self.bias = None

    def compute_kernel_bias(self, sentence_vec):
        '''bert-whitening的torch实现
        '''
        vecs = torch.cat(sentence_vec, dim=0)
        self.bias = -vecs.mean(dim=0, keepdims=True)

        cov = torch.cov(vecs.T)  # 协方差
        u, s, vh = torch.linalg.svd(cov)
        W = torch.matmul(u, torch.diag(s**0.5))
        self.kernel = torch.linalg.inv(W.T)
    
    def save_whiten(self, path):
        whiten = {'kernel': self.kernel, 'bias': self.bias}
        torch.save(path, whiten)
        
    def load_whiten(self, path):
        whiten = torch.load(path)
        self.kernel = whiten['kernel']
        self.bias = whiten['bias']

    def transform_and_normalize(self, vecs):
        """应用变换，然后标准化
        """
        if not (self.kernel is None or self.bias is None):
            vecs = (vecs + self.bias).mm(self.kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        '''
        sequence_output = self.dense(inputs)  # [..., heads*head_size*2]
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2)  # [..., heads, head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.head_size**0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        '''
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2) / 2 # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class TplinkerHandshakingKernel(nn.Module):
    '''Tplinker的HandshakingKernel实现
    '''
    def __init__(self, hidden_size, shaking_type, inner_enc_type=''):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        
        # 自行实现的用torch.gather方式来做，避免循环，目前只实现了cat方式
        # tag_ids = [(i, j) for i in range(maxlen) for j in range(maxlen) if j >= i]
        # gather_idx = torch.tensor(tag_ids, dtype=torch.long).flatten()[None, :, None]
        # self.register_buffer('gather_idx', gather_idx)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens

        # def handshaking_kernel(self, last_hidden_state):
        #     '''获取(0,0),(0,1),...,(99,99))对应的序列id
        #     '''
        #     btz, _, hdsz = last_hidden_state.shape
        #     gather_idx = self.gather_idx.repeat(btz, 1, hdsz)
        #     concat_hidden_states = torch.gather(last_hidden_state, dim=1, index=gather_idx)  # [btz, pair_len*2, hdsz]
        #     concat_hidden_states = concat_hidden_states.reshape(btz, -1, 2, hdsz)  # concat方式 [btz, pair_len, 2, hdsz]
        #     shaking_hiddens = torch.cat(torch.chunk(concat_hidden_states, chunks=2, dim=-2), dim=-1).squeeze(-2)  # [btz, pair_len, hdsz*2]
        #     return shaking_hiddens


class MixUp(nn.Module):
    '''mixup方法实现
    
    :param method: str, 可选'embed', ’encoder‘分别表示在embedding和encoder层面做mixup, None表示mix后续处理, ’hidden‘表示对隐含层做mixup
    :param alpha: float
    :param layer_mix: None or int, 需要mix的隐含层index
    '''
    def __init__(self, method='encoder', alpha=1.0, layer_mix=None):
        super().__init__()
        assert method in {'embed', 'encoder', 'hidden', None}
        self.method = method
        self.alpha = alpha
        self.perm_index = None
        self.lam = 0
        self.layer_mix = layer_mix  # 需要mix的隐含层index
    
    def get_perm(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs[self.perm_index]
        elif isinstance(inputs, (list, tuple)):
            return [inp[self.perm_index] if isinstance(inp, torch.Tensor) else inp for inp in inputs]
    
    def mix_up(self, output, output1):
        if isinstance(output, torch.Tensor):
            return self.lam * output + (1.0-self.lam) * output1
        elif isinstance(output, (list, tuple)):
            output_final = []
            for i in range(len(output)):
                if output[i] is None: # conditional_emb=None
                    output_final.append(output[i])
                elif (not output[i].requires_grad) and (output[i].dtype in {torch.long, torch.int}):
                    # 不是embedding形式的
                    output_final.append(torch.max(output[i], output1[i]))
                else:
                    output_final.append(self.lam * output[i] + (1.0-self.lam) * output1[i])
            return output_final
        else:
            raise ValueError('Illegal model output')

    def encode(self, model, inputs):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        self.lam = np.random.beta(self.alpha, self.alpha)
        self.perm_index = torch.randperm(batch_size).to(device)

        if self.method is None:
            output = model(inputs)
            output1 = self.get_perm(output)
            return [output, output1]

        elif self.method == 'encoder':
            output = model(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)

        elif self.method == 'embed':
            output = model.apply_embeddings(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
            # Main
            output_final = model.apply_main_layers(output_final)
            # Final
            output_final = model.apply_final_layers(output_final)
        
        elif self.method == 'hidden':
            if self.layer_mix is None:
                # 这里暂时只考虑encoderLayer, 不考虑decoderLayer和seq2seq模型结构
                try:
                    layer_mix = random.randint(0, len(model.encoderLayer))
                except:
                    warnings.warn('LayerMix random failded')
                    layer_mix = 0
            else:
                layer_mix = self.layer_mix
            
            def apply_on_layer_end(l_i, output):
                if l_i == layer_mix:
                    output1 = self.get_perm(output)
                    return self.mix_up(output, output1)
                else:
                    return output
            model.apply_on_layer_end = apply_on_layer_end
            output_final = model(inputs)
        return output_final
    
    def forward(self, criterion, y_pred, y_true):
        '''计算loss
        '''
        y_true1 = y_true[self.perm_index]
        return self.lam * criterion(y_pred, y_true) + (1 - self.lam) * criterion(y_pred, y_true1)


class ConvLayer(nn.Module):
    '''deberta_v2中使用
    '''
    def __init__(self, hidden_size, dropout_rate=0.1, layer_norm_eps=1e-12, conv_kernel_size=3, conv_groups=1, conv_act="tanh", **kwargs):
        super().__init__()
        kernel_size = conv_kernel_size
        groups = conv_groups
        self.conv_act = conv_act
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)  # 这里使用bert4torch的LayerNorm会有问题
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = get_activation(self.conv_act)(self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        return output_states


class MultiSampleDropout(nn.Module):
    """multisample dropout (wut): https://arxiv.org/abs/1905.09788
    """
    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits