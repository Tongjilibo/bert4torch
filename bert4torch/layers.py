import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from bert4torch.snippets import get_sinusoid_encoding_table
from bert4torch.activations import get_activation


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
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, attention_scale=True,
                 return_attention_scores=False, bias=True, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.bias = bias
        self.q = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o = nn.Linear(hidden_size, hidden_size, bias=bias)
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # hidden_states shape: [batch_size, seq_q, hidden_size]
        # attention_mask shape: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        # encoder_hidden_states shape: [batch_size, seq_k, hidden_size]
        # encoder_attention_mask shape: [batch_size, 1, 1, seq_k]

        mixed_query_layer = self.q(hidden_states)
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
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
        if attention_mask is not None:
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
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

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)


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
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512)).to(mask), dim=dim)
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
    """
        embeddings层
        构造word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, max_position, segment_vocab_size, shared_segment_embeddings, drop_rate, conditional_size=False, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # 位置编码
        if kwargs.get('p_bias') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)
        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'other_relative'}:
            # 如果使用相对位置编码，则不声明PositionEmbeddings
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        
        # segement编码
        if (segment_vocab_size > 0) and (not shared_segment_embeddings):
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        # emb_scale
        self.emb_scale = kwargs.get('emb_scale', 1)  # transform_xl, xlnet特有

        # LayerNorm
        self.layerNorm = LayerNorm(embedding_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.dropout = nn.Dropout(drop_rate)

        # 如果embedding_size != hidden_size，则再有一个linear(适用于albert矩阵分解)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def forward(self, token_ids, segment_ids=None, conditional_emb=None, additional_embs=None):
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
            position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(token_ids.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale  # transform_xl, xlnet特有

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm((embeddings, conditional_emb))
        embeddings = self.dropout(embeddings)

        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
        Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

        注意: 1、以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
              2、原始的Transformer的encoder中的Feed Forward层一共有两层linear，
              config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, 
                 is_dropout=False, conditional_size=False, **kwargs):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act, is_dropout=is_dropout, **kwargs)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.is_decoder = kwargs.get('is_decoder')
        if self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, **kwargs)
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
                self.seg_embed = nn.Embedding(segment_vocab_size, self.hidden_size)

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
            if hasattr(self, 'seg_embed'):
                w_head_s = self.seg_embed(seg_mat)  # [btz, q_len, klen, hdsz]
                w_head_s = w_head_s.reshape(*w_head_s.shape[:3], self.num_attention_heads, self.attention_head_size)

            r_head_k = self.r(r)  # [hdsz, nhead*headsize] = [r_len, 1, nhead*headsize]
            r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)  # rlen x n_head x d_head

            #### compute attention score
            rw_head_q = w_head_q + self.r_w_bias.unsqueeze(1)  # [btz, n_head, q_len, d_head]
            AC = torch.einsum('bnid,bnjd->bnij', (rw_head_q, w_head_k))  # [btz, n_head, q_len, k_len]

            rr_head_q = w_head_q + self.r_r_bias.unsqueeze(1)  # [btz, n_head, q_len, d_head]
            BD = torch.einsum('bnid,jnd->bnij', (rr_head_q, r_head_k))  # [btz, n_head, q_len, k_len]
            BD = self.rel_shift_bnij(BD, klen=AC.shape[3]) if self.rel_shift_opt == 'xlnet' else self.rel_shift(BD)

            if hasattr(self, 'seg_embed') and (self.r_r_bias is not None):
                rs_head_q = w_head_q + self.r_s_bias.unsqueeze(1)
                EF = torch.einsum('bnid,bijnd->bnij', (rs_head_q, w_head_s))  # [btz, n_head, q_len, k_len]
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
        # position_embeddings = torch.take_along_dim(embeddings_table, final_mat.flatten().unsqueeze(1), dim=0)
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
    '''直接从pytorch版本的bert中移植过来的
    '''
    def __init__(self, num_labels, init_transitions=None, freeze=False):
        super(CRF, self).__init__()
        self.num_labels = num_labels
        self.START_TAG_IDX = -2
        self.END_TAG_IDX = -1
        if init_transitions is None:
            init_transitions = torch.zeros(self.num_labels + 2, self.num_labels + 2)
        else:
            assert init_transitions.shape == (self.num_labels + 2, self.num_labels + 2), 'CRF init_weight shape does not match'
            init_transitions = torch.tensor(init_transitions, dtype=torch.float)
        init_transitions[:, self.START_TAG_IDX] = -10000.0
        init_transitions[self.END_TAG_IDX, :] = -10000.0
        
        if not freeze:
            self.transitions = nn.Parameter(init_transitions)
        else:
            self.register_buffer('transitions', init_transitions)

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    def _forward_alg(self, feats, mask):
        bts, seq_len, tag_size = feats.size()
        ins_num = bts * seq_len

        mask = mask.transpose(1, 0).contiguous()  # [seq_len, bsz]

        # [seq_len * bts, tag_size, tag_size]
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # [seq_len * bts, tag_size, tag_size]
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, bts, tag_size, tag_size)  # [seq_len, bts, tag_size, tag_size]
        seq_iter = enumerate(scores)

        """ only need start from start_tag """
        _, inivalues = next(seq_iter)  # [bts, tag_size, tag_size]
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(bts, tag_size, 1)  # [bts, tag_size, 1]

        for idx, cur_values in seq_iter:  # scalar, [bts, tag_size, tag_size]
            # [bts, tag_size, tag_size]
            cur_values = cur_values + partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
            cur_partition = self.log_sum_exp(cur_values, tag_size)  # [bts, tag_size]
            mask_idx = mask[idx, :].view(bts, 1).expand(bts, tag_size)  # [bts, tag_size]
            """ effective updated partition part, only keep the partition value of mask value = 1 """
            masked_cur_partition = cur_partition.masked_select(mask_idx.bool())  # [x * tag_size]
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(bts, tag_size, 1)  # [bts, tag_size, 1]
                """ replace the partition where the maskvalue=1, other partition value keeps the same """
                partition.masked_scatter_(mask_idx.bool(), masked_cur_partition)
        # [bts, tag_size, tag_size]
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(bts, tag_size, tag_size) + \
                     partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
        cur_partition = self.log_sum_exp(cur_values, tag_size)  # [bts, tag_size]
        final_partition = cur_partition[:, self.END_TAG_IDX]  # [bts]
        return final_partition.sum(), scores

    # scores: [seq_len, bts, tag_size, tag_size]
    # mask: [bts, seq_len]
    # tags: [bts, seq_len]
    def _score_sentence(self, scores, mask, tags):
        seq_len, btz, tag_size, _ = scores.size()

        """ convert tag value into a new format, recorded label bigram information to index """
        new_tags = torch.empty(btz, seq_len, requires_grad=True).to(tags)  # [btz, seq_len]
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]  # `tag_size - 2` account for `START_TAG_IDX`
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, btz, 1)  # [seq_len, btz, 1]

        # get all energies except end energy
        tg_energy = torch.gather(scores.view(seq_len, btz, -1), 2, new_tags).view(seq_len, btz)  # [seq_len, btz]
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0).bool())  # list

        """ transition for label to STOP_TAG """
        # [btz, tag_size]
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(1, tag_size).expand(btz, tag_size)
        """ length for batch,  last word position = length - 1 """
        length_mask = torch.sum(mask, dim=1, keepdim=True).long()  # [bts, 1]
        """ index the label id of last word """
        end_ids = torch.gather(tags, 1, length_mask - 1)  # [bts, 1]
        """ index the transition score for end_id to STOP_TAG """
        end_energy = torch.gather(end_transition, 1, end_ids)  # [bts, 1]

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    # tags: [bts, seq_len]
    def neg_log_likelihood_loss(self, feats, mask, tags):
        bts = feats.size(0)
        # scalar, [seq_len, bts, tag_size, tag_size]
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return (forward_score - gold_score) / bts

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    def _viterbi_decode(self, feats, mask):
        bts, seq_len, tag_size = feats.size()
        ins_num = seq_len * bts
        """ calculate sentence length for each sentence """
        length_mask = torch.sum(mask, dim=1, keepdim=True).long()  # [bts, 1]
        mask = mask.transpose(1, 0).contiguous()  # [seq_len, bts]

        # [seq_len * bts, tag_size, tag_size]
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # [seq_len * bts, tag_size, tag_size]
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, bts, tag_size, tag_size)  # [seq_len, bts, tag_size, tag_size]

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = []
        partition_history = []
        mask = (1 - mask.long()).bool()  # [seq_len, bts]
        _, inivalues = next(seq_iter)  # [bts, tag_size, tag_size]
        """ only need start from start_tag """
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(bts, tag_size, 1)  # [bts, tag_size,1]
        partition_history.append(partition)

        for idx, cur_values in seq_iter:  # scalar, [bts, tag_size, tag_size]
            # [bts, tag_size, tag_size]
            cur_values = cur_values + partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
            """ do not consider START_TAG/STOP_TAG """
            partition, cur_bp = torch.max(cur_values, 1)  # [bts, tag_size], [bts, tag_size]
            partition_history.append(partition.unsqueeze(-1))
            """ set padded label as 0, which will be filtered in post processing"""
            cur_bp.masked_fill_(mask[idx].view(bts, 1).expand(bts, tag_size), 0)  # [bts, tag_size]
            back_points.append(cur_bp)

        # [bts, seq_len, tag_size]
        partition_history = torch.cat(partition_history).view(seq_len, bts, -1).transpose(1, 0).contiguous()
        """ get the last position for each setences, and select the last partitions using gather() """
        last_position = length_mask.view(bts, 1, 1).expand(bts, 1, tag_size) - 1  # [bts, 1, tag_size]
        # [bts, tag_size, 1]
        last_partition = torch.gather(partition_history, 1, last_position).view(bts, tag_size, 1)
        """ calculate the score from last partition to end state (and then select the STOP_TAG from it) """
        # [bts, tag_size, tag_size]
        last_values = last_partition.expand(bts, tag_size, tag_size) + \
                      self.transitions.view(1, tag_size, tag_size).expand(bts, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)  # [bts, tag_size]
        """ select end ids in STOP_TAG """
        pointer = last_bp[:, self.END_TAG_IDX]  # [bts]

        pad_zero = torch.zeros(bts, tag_size, requires_grad=True).to(mask).long()  # [bts, tag_size]
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, bts, tag_size)  # [seq_len, bts, tag_size]

        insert_last = pointer.contiguous().view(bts, 1, 1).expand(bts, 1, tag_size)  # [bts, 1, tag_size]
        back_points = back_points.transpose(1, 0).contiguous()  # [bts, seq_len, tag_size]
        """move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values """
        back_points.scatter_(1, last_position, insert_last)  # [bts, seq_len, tag_size]

        back_points = back_points.transpose(1, 0).contiguous()  # [seq_len, bts, tag_size]
        """ decode from the end, padded position ids are 0, which will be filtered if following evaluation """
        decode_idx = torch.empty(seq_len, bts, requires_grad=True).to(pointer)  # [seq_len, bts]
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bts, 1))
            decode_idx[idx] = pointer.view(-1).data
        decode_idx = decode_idx.transpose(1, 0)  # [bts, seq_len]
        return decode_idx

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    def forward(self, feats, mask):
        best_path = self._viterbi_decode(feats, mask)  # [bts, seq_len]
        return best_path
    
    @staticmethod
    def log_sum_exp(vec, m_size):
        _, idx = torch.max(vec, 1)  # B * 1 * M
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
        return max_score.view(-1, m_size) + torch.log(torch.sum(
            torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


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
        self.RoPE = RoPE

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
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
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.RoPE = RoPE

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
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
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, heads, seq_len, 2]
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
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
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