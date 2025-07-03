from torch import nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from bert4torch.layers.position_encoding import (
    DebertaV2PositionsEncoding, 
    NezhaPositionsEncoding, 
    T5PositionsEncoding, 
    RopePositionEncoding, 
    RopeLinearScalingPositionEncoding,
    RopeDynamicNTKScalingPositionEncoding,
    RopeDynamicNTKScalingQwenPositionEncoding,
    RopeLlama3PositionEncoding,
    RopeYarnPositionEncoding,
    ROPE_ENCODGING_MAP,
    ALiBiPositionsEncoding
)
from bert4torch.layers.core import LayerNorm
from bert4torch.activations import get_activation
from bert4torch.snippets import log_warn_once, is_xformers_available
from bert4torch.layers.attention.attention_utils import eager_attention_forward, sdpa_attention_forward, flash_attention_forward
from typing import Literal, Optional, Tuple, Union
import copy


if is_xformers_available():
    from xformers import ops as xops


class MultiHeadAttention(nn.Module):
    '''多头注意力
    :param hidden_size: int, 隐含层神经元个数
    :param num_attention_heads: int, 多头注意力的多头数
    :param attention_probs_dropout_prob: float，softmax后的dropout rate
    :param dropout_rate: float, pos_dropout对应的dropout rate, 目前仅在deverta中使用，默认为0.1
    :param attention_scale: bool, 是否对attention_scores进行缩放，默认为True
    :param output_attentions: bool，是否返回attention_scores，默认为False
    :param bias: bool, qkvo的weight是否包含bias，默认为True
    :param rope_scaling: dict, rope的position encoding的参数，默认为None
    :param _attn_implementation: Literal枚举值，计算attention score的方式，支持'sdpa', 'xformers', 'flash_attn_2', "eager"等, 默认为None
    :param use_logn_attn: bool，是否使用use_logn_attn, 默认为None
    :param layer_idx: int，transformer block的层序号
    '''
    def __init__(self, 
                 hidden_size:int, 
                 num_attention_heads:int, 
                 attention_probs_dropout_prob:float, 
                 dropout_rate:float=0.1, 
                 attention_scale:bool=True,
                 output_attentions:bool=False, 
                 bias:bool=True, 
                 rope_scaling:dict=None, 
                 _attn_implementation:Literal['sdpa', 'xformers', 'flash_attn_2', 'eager']='eager', 
                 use_logn_attn:bool=None, 
                 layer_idx:int=None,
                 num_key_value_heads:int=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.dropout_rate = dropout_rate
        self.is_decoder = kwargs.get('is_decoder', False)
        self.attention_scale = attention_scale
        self.output_attentions = output_attentions
        self.bias = bias
        self.rope_scaling = rope_scaling or dict()
        self.layer_idx = layer_idx
        self.sliding_window = kwargs.get('sliding_window')
        self.max_window_layers = kwargs.get('max_window_layers')
        self._attn_implementation = _attn_implementation  # attention的实现
        self.use_logn_attn = use_logn_attn # 使用logn_attn
        self.max_position = kwargs.get('max_position')
        # t5_pegasus_small中hidden_size/num_attention_heads != 0
        # 苏神的roberta small中qk的维度和v不同
        self.attention_head_size = kwargs.get('attention_head_size', int(hidden_size/num_attention_heads))  # Attention中V的head_size
        self.attention_key_size = kwargs.get('attention_key_size', self.attention_head_size)  # Attention中Q,K的head_size
        self.scaling = self.attention_head_size ** (-0.5)
        q_inner_dim = k_inner_dim = self.attention_key_size * num_attention_heads
        v_inner_dim = self.attention_head_size * num_attention_heads

        # multi query attention: chatglm中叫num_key_value_heads
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
            k_inner_dim_tmp = self.attention_head_size * self.num_key_value_heads
            v_inner_dim_tmp = k_inner_dim_tmp

        # longlora
        if kwargs.get('longlora_group_size') is not None:
            self.longlora_group_size = kwargs.get('longlora_group_size')

        self.q = nn.Linear(hidden_size, q_inner_dim, bias=bias)
        self.k = nn.Linear(hidden_size, k_inner_dim_tmp if hasattr(self, 'num_key_value_heads') else k_inner_dim, bias=bias)
        self.v = nn.Linear(hidden_size, v_inner_dim_tmp if hasattr(self, 'num_key_value_heads') else v_inner_dim, bias=bias)
        self.o = nn.Linear(v_inner_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob) if attention_probs_dropout_prob > 0 else lambda x: x
        self.init_position_encoding(**kwargs)

    def init_position_encoding(self, **kwargs):
        '''初始化相对位置编码'''
        pass

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        '''获取qkv states，主要是为了下游继承'''
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        if (encoder_hidden_states is not None) and (past_key_value is not None):
            key_states, value_states = past_key_value
            attention_mask = encoder_attention_mask
        elif encoder_hidden_states is not None:
            key_states = self.transpose_for_k_scores(self.k(encoder_hidden_states))
            value_states = self.transpose_for_v_scores(self.v(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            value_states = self.transpose_for_v_scores(self.v(hidden_states))
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            value_states = self.transpose_for_v_scores(self.v(hidden_states))
        return query_states, key_states, value_states, attention_mask

    def forward(self, 
                hidden_states:Optional[torch.Tensor]=None, 
                attention_mask:Optional[torch.FloatTensor]=None, 
                encoder_hidden_states:Optional[torch.FloatTensor]=None, 
                encoder_attention_mask:Optional[torch.FloatTensor]=None, 
                past_key_value:Optional[Tuple[Tuple[torch.FloatTensor]]]=None, 
                position_ids=None, 
                **model_kwargs
        ):
        '''
        :param hidden_states: [batch_size, seq_q, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        :param encoder_hidden_states: [batch_size, seq_k, hidden_size]
        :param encoder_attention_mask: [batch_size, 1, 1, seq_k]
        :param past_key_value: ([batch_size, num_attention_heads, key_len_cache, attention_head_size], ...)
        '''
        query_states, key_states, value_states, attention_mask = self._get_qkv_states(
            hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids)
        # query_states shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_states shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_states shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        # 使用logn_attn
        if self.use_logn_attn:
            query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(self.max_position)).clip(1).to(query_states.dtype)

        # past_key_values
        if self.is_decoder and (not self.training):  # 仅推理是记录
            past_key_value = (key_states, value_states)

        # multi_query_attention
        if hasattr(self, 'num_key_value_heads') and self.num_key_value_heads > 1:
            key_states = self.repeat_kv(key_states)
            value_states = self.repeat_kv(value_states)

        # longlora
        if hasattr(self, 'longlora_group_size'):
            query_states, key_states, value_states, attention_mask = self.longlora_shift(query_states, key_states, value_states, attention_mask)


        # ====================================attention的多类实现====================================
        context_layer, attention_scores = self.attention_forward(query_states, key_states, value_states, attention_mask, past_key_value)

        if hasattr(self, 'longlora_group_size'):  # context_layer: [bsz * (q_len // group_size), num_heads, group_size, head_dim]
            bsz, q_len = hidden_states.shape[:2]
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.reshape(bsz, q_len, self.num_attention_heads, self.attention_head_size)
            # shift back
            context_layer[:, :, self.num_attention_heads//2:] = context_layer[:, :, self.num_attention_heads//2:].roll(self.longlora_group_size//2, dims=1)
            context_layer = context_layer.reshape(bsz, q_len, self.hidden_size)
        else:
            # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size()[-2]*context_layer.size()[-1],)
            context_layer = context_layer.reshape(*new_context_layer_shape).contiguous()

        # 是否返回attention scores
        outputs = (self.o(context_layer), attention_scores) if self.output_attentions else (self.o(context_layer),)
        return outputs + (past_key_value,) if self.is_decoder else outputs
    
    def attention_forward(self, query_states:torch.FloatTensor, key_states:torch.FloatTensor, value_states:torch.FloatTensor, 
                          attention_mask:torch.Tensor, past_key_value:Optional[Tuple[Tuple[torch.FloatTensor]]]=None, **kwargs):
        '''各类attention的实现, 方便继承'''
        if (self._attn_implementation == 'xformers') and self.training:
            # xformers
            context_layer = xops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask())
            attention_scores = None
        elif self._attn_implementation in {True, 'sdpa'}:
            # SDPA
            context_layer, attention_scores = sdpa_attention_forward(self, query_states, key_states, value_states, attention_mask)
        elif self._attn_implementation == 'flash_attn_2':
            # flash_attn
            context_layer, attention_scores = flash_attention_forward(self, query_states, key_states, value_states, attention_mask, 
                                                                      past_key_value=past_key_value)
        if self._attn_implementation in {None, 'eager'}:
            # torch原生实现
            context_layer, attention_scores = eager_attention_forward(self, query_states, key_states, value_states, attention_mask)
        return context_layer, attention_scores

    def repeat_kv(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(2)
        hidden_states = hidden_states.expand(-1, -1, self.num_attention_heads // self.num_key_value_heads, -1, -1)
        hidden_states = hidden_states.contiguous().view(hidden_states.shape[:1] + (self.num_attention_heads,) + hidden_states.shape[-2:])
        return hidden_states

    def longlora_shift(self, query_states, key_states, value_states, attention_mask):
        '''longlora中对qkv和mask进行修改: https://github.com/dvlab-research/LongLoRA'''
        # query_states shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_states shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_states shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv

        bsz, _, q_len, _ = query_states.shape
        num_group = q_len // self.longlora_group_size
        query_states = shift(query_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        key_states = shift(key_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        value_states = shift(value_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        attention_mask = attention_mask[:, :, :self.longlora_group_size, :self.longlora_group_size].repeat(num_group, 1, 1, 1)
        # qkv: [bsz * (q_len // group_size), num_heads, group_size, head_dim]
        return query_states, key_states, value_states, attention_mask

    def transpose_for_q_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_key_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_k_scores(self, x):
        if hasattr(self, 'num_key_value_heads'):
            new_x_shape = x.size()[:-1] + (self.num_key_value_heads, self.attention_key_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_key_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_v_scores(self, x):
        if hasattr(self, 'num_key_value_heads'):
            new_x_shape = x.size()[:-1] + (self.num_key_value_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
   
    def apply_attention_scale(self, attention_scores):
        '''方便子类继承'''
        return attention_scores * self.scaling
    
    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        return attention_scores


class DebertaV2Attention(MultiHeadAttention):
    def init_position_encoding(self, **kwargs):
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
        self.relative_positions = DebertaV2PositionsEncoding(qlen=self.max_position, 
                                                                klen=self.max_position, 
                                                                position_buckets=kwargs.get('position_buckets'),
                                                                max_position=self.max_position)
        self.relative_positions_encoding = nn.Embedding(self.max_position, self.hidden_size)
        self.norm_rel_ebd = [x.strip() for x in kwargs.get("norm_rel_ebd", "none").lower().split("|")]
        if "layer_norm" in self.norm_rel_ebd:
            self.layernorm = nn.LayerNorm(self.hidden_size, kwargs.get('layer_norm_eps', 1e-12), elementwise_affine=True)
        self.pos_dropout = nn.Dropout(self.dropout_rate)

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):
            return attention_scores
        
        # ==================== deberta_v2相对位置编码 ====================
        self.attention_scale = False  # deberta_v2使用自己的attention_scale
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_states.size(-1), dtype=torch.float) * scale_factor)
        attention_scores = attention_scores / scale.to(dtype=query_states.dtype)

        rel_embeddings = self.pos_dropout(self.layernorm(self.relative_positions_encoding.weight))
        relations_keys = self.relative_positions(attention_scores.shape[-2], attention_scores.shape[-1])
        rel_att = self.apply_deberta_pos_emb(query_states, key_states, relations_keys, rel_embeddings, scale_factor)
        attention_scores = attention_scores + rel_att
        return attention_scores

    def apply_deberta_pos_emb(self, query_states:torch.FloatTensor, key_states:torch.FloatTensor, relative_pos, rel_embeddings, scale_factor):
        '''deberta_v2使用，和原版区别是query_states是4维, 原disentangled_attention_bias'''
        btz, n_head, q_len, d_head = query_states.size()
        k_len = key_states.size(-2)
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
        relative_pos = relative_pos.long().to(query_states.device)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_states = self.transpose_for_q_scores(self.q(rel_embeddings)).repeat(btz, 1, 1, 1)
            pos_key_states = self.transpose_for_k_scores(self.k(rel_embeddings)).repeat(btz, 1, 1, 1)
        else:
            # 这里逻辑去掉了
            pass

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            c2p_att = torch.matmul(query_states, pos_key_states.transpose(-1, -2))
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
            p2c_att = torch.matmul(key_states, pos_query_states.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([btz, n_head, k_len, k_len])).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)
        return score


class AlibiAttention(MultiHeadAttention):
    '''alibi相对位置编码'''
    def init_position_encoding(self, **kwargs):
        self.relative_positions_encoding = ALiBiPositionsEncoding(self.num_attention_heads)
    
    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        attention_scores = self.apply_alibi_pos_emb(attention_scores, key_states)
        return attention_scores
    
    def apply_alibi_pos_emb(self, attention_scores, key_states):
        ''' 执行alibi相对位置编码，单独拎出来主要是falcon是在+之后再执行attention_scale的 '''
        attention_scores = self.apply_attention_scale(attention_scores)
        key_position_scores_r_t = self.relative_positions_encoding(key_states)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = torch.max(attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min))  # baichuan-13b逻辑
        self.attention_scale = False
        return attention_scores


class NezhaTypicalRelativeAttention(MultiHeadAttention):
    def init_position_encoding(self, **kwargs):
        self.relative_positions_encoding = NezhaPositionsEncoding(
            qlen=self.max_position, 
            klen=self.max_position,
            embedding_size=self.attention_head_size,
            max_relative_position=kwargs.get('max_relative_position')
            )
        
    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):
            return attention_scores
        
        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        # ==================== nezha相对位置编码 ====================
        relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])  # [to_seq_len, to_seq_len, d_hid]
        # 旧实现，方便读者理解维度转换
        # query_states_t = query_states.permute(2, 0, 1, 3)
        # query_states_r = query_states_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, self.attention_head_size)
        # key_position_scores = torch.matmul(query_states_r, relations_keys.permute(0, 2, 1))
        # key_position_scores_r = key_position_scores.view(from_seq_length, batch_size, num_attention_heads, from_seq_length)
        # key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        # 新实现
        key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_states, relations_keys)
        attention_scores = attention_scores + key_position_scores_r_t
        return attention_scores

    def attention_forward(self, query_states:torch.FloatTensor, key_states:torch.FloatTensor, value_states:torch.FloatTensor, 
                          attention_mask:torch.Tensor, *args, **kwargs):
        '''qkv attention: torch原生实现'''
        output = eager_attention_forward(self, query_states, key_states, value_states, attention_mask, 
                                         return_dict_name=['context_layer', 'attention_scores', 'attention_probs'])

        # ==================== nezha相对位置编码 ====================
        relations_values = self.relative_positions_encoding(output['attention_scores'].shape[-1], output['attention_scores'].shape[-1])
        # 旧实现，方便读者理解维度转换
        # attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        # attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, to_seq_length)
        # value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        # value_position_scores_r = value_position_scores.view(from_seq_length, batch_size, num_attention_heads, self.attention_head_size)
        # value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        # 新实现
        value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', output['attention_probs'], relations_values)
        context_layer = output['context_layer'] + value_position_scores_r_t

        return context_layer, output['attention_scores']


class RopeAttention(MultiHeadAttention):
    def init_position_encoding(self, **kwargs):
        rope_scaling = copy.deepcopy(self.rope_scaling)
        scaling_type = rope_scaling.pop("rope_type", rope_scaling.pop('type', None))
        scaling_factor = rope_scaling.pop("factor", None)
        rope_theta = kwargs.get('rope_theta')
        rope_rank = kwargs.get('rope_rank')
        if scaling_type is None:
            assert scaling_factor is None , f'Args `rope_scaling.factor` not supported in standard rope'
        elif scaling_type in {'linear', 'dynamic'}:
            assert scaling_factor is not None and scaling_factor != 1, f'Args `rope_scaling.factor`={scaling_factor} which is illegal'
        
        self.relative_positions_encoding = ROPE_ENCODGING_MAP[scaling_type](
            embedding_size=self.attention_head_size, 
            max_position=self.max_position, 
            max_seq_len_cached=kwargs.get('rope_max_seq_len_cached', self.max_position),
            sin_cos_cached = kwargs.get('rope_sin_cos_cached', False),
            rope_rank=rope_rank, 
            scaling_factor=scaling_factor, 
            rope_theta=rope_theta,
            **rope_scaling)

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        key_states = self.transpose_for_k_scores(self.k(hidden_states))
        value_states = self.transpose_for_v_scores(self.v(hidden_states))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # 执行相对位置编码
        query_states, key_states = self.relative_positions_encoding([query_states, key_states], position_ids)
        
        # rotary有cache情况下，需要先rope后再和past_key_value concat
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        return query_states, key_states, value_states, attention_mask


class Qwen3Attention(RopeAttention):
    '''qwen3的注意力机制
        - 有q_norm和k_norm
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-6)
        self.q_norm = LayerNorm(self.attention_head_size, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)
        self.k_norm = LayerNorm(self.attention_key_size, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)

    def transpose_for_q_scores(self, x):
        return self.q_norm(super().transpose_for_q_scores(x))

    def transpose_for_k_scores(self, x):
        return self.k_norm(super().transpose_for_k_scores(x))
    

class GatedAttention(nn.Module):
    '''门控注意力单元
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
        
        self.p_bias = kwargs.get('p_bias')
        if self.p_bias == 'rotary':  # RoPE
            self.relative_positions_encoding = RopePositionEncoding(embedding_size=self.attention_head_size, **kwargs)

    def forward(self, hidden_states, attention_mask, position_ids):
        # 投影变换
        hidden_states = self.hidden_fn(self.i_dense(hidden_states))
        u, v, qk = hidden_states.split([self.intermediate_size, self.intermediate_size, self.attention_head_size], dim=-1)
        q, k = self.offsetscale(qk)  # 仿射变换

        # 加入RoPE
        if self.p_bias == 'rotary':
            q = self.relative_positions_encoding(q, position_ids)
            k = self.relative_positions_encoding(k, position_ids)

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
        '''仿射变换'''
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


class TransformerxlMultiHeadAttn(MultiHeadAttention):
    '''Transformer_XL式相对位置编码RelPartialLearnableMultiHeadAttn, 这里修改成了MultiHeadAttention的batch_first代码格式'''
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
        '''transformer_xl使用, 向左shift让右上角都是0, 对角线是同一个值, x: [btz, n_head, q_len, k_len]'''
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
        ''' xlnet使用'''
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

        w_head_q = self.transpose_for_q_scores(mixed_query_layer)  # [btz, n_head, q_len, d_head]
        w_head_k = self.transpose_for_k_scores(mixed_key_layer)  # [btz, n_head, k_len, d_head]
        w_head_v = self.transpose_for_v_scores(mixed_value_layer)  # [btz, n_head, k_len, d_head]

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
        outputs = (self.o(context_layer), attention_scores) if self.output_attentions else (self.o(context_layer),)
        return outputs


class DeepseekV2Attention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_lora_rank = kwargs.get('q_lora_rank')
        self.kv_lora_rank = kwargs.get('kv_lora_rank')
        self.qk_nope_head_dim = kwargs.get('qk_nope_head_dim')
        self.qk_rope_head_dim = kwargs.get('qk_rope_head_dim')
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-6)
        if self.q_lora_rank is None:
            self.q = nn.Linear(self.hidden_size, self.num_attention_heads * self.q_head_dim, bias=self.bias)
        else:
            del self.q
            self.q_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=self.bias)
            self.q_a_layernorm = LayerNorm(self.q_lora_rank, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)
            self.q_b = nn.Linear(self.q_lora_rank, self.attention_key_size * self.q_head_dim, bias=self.bias)

        del self.k, self.v
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=self.bias)
        self.kv_a_layernorm = LayerNorm(self.kv_lora_rank, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)
        self.kv_b = nn.Linear(self.kv_lora_rank, self.num_attention_heads * 
                              (self.q_head_dim - self.qk_rope_head_dim + self.attention_head_size), bias=self.bias)
        self.o = nn.Linear(self.num_attention_heads * self.attention_head_size, self.hidden_size, bias=self.bias)

        self.scaling = self.q_head_dim ** (-0.5)
        if self.rope_scaling is not None:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = 1.0 if scaling_factor <= 1 else 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scaling = self.scaling * mscale * mscale
        
    def init_position_encoding(self, **kwargs):
        '''这里dim为qk_rope_head_dim所以重新初始化了'''
        rope_scaling = copy.deepcopy(self.rope_scaling)
        scaling_type = rope_scaling.pop("rope_type", rope_scaling.pop('type', None))
        scaling_factor = rope_scaling.pop("factor", None)
        rope_theta = kwargs.get('rope_theta')
        rope_rank = kwargs.get('rope_rank')
        self.relative_positions_encoding = ROPE_ENCODGING_MAP[scaling_type](
            embedding_size = kwargs.get('qk_rope_head_dim'), 
            max_position = self.max_position, 
            max_seq_len_cached=kwargs.get('rope_max_seq_len_cached', self.max_position),
            sin_cos_cached = kwargs.get('rope_sin_cos_cached', False),
            rope_rank = rope_rank, 
            scaling_factor = scaling_factor, 
            rope_theta = rope_theta,
            **rope_scaling
            )
    
    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q(hidden_states)
        else:
            q = self.q_b(self.q_a_layernorm(self.q_a(hidden_states)))
        q = q.view(bsz, q_len, self.num_attention_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_attention_heads, self.qk_nope_head_dim + self.attention_head_size)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.attention_head_size], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        q_pe, k_pe = self.relative_positions_encoding([q_pe, k_pe], position_ids)
        k_pe : torch.Tensor
        query_states = k_pe.new_empty(bsz, self.num_attention_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_attention_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        # 过了rope再concat
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        return query_states, key_states, value_states, attention_mask
    

class T5Attention(MultiHeadAttention):
    def init_position_encoding(self, **kwargs):
        self.relative_positions = T5PositionsEncoding(
            qlen=self.max_position,  
            klen=self.max_position, 
            relative_attention_num_buckets=kwargs.get('relative_attention_num_buckets'), 
            is_decoder=kwargs.get('is_decoder'))
        self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'), self.num_attention_heads)
    
    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):  # 外部可能会变更
            return attention_scores

        # ==================== t5相对位置编码 ====================
        relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])  # 都是klen, 应对use_state=True
        key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
        key_position_scores_r_t = key_position_scores_r_t[:, :, -attention_scores.shape[-2] :, :]  # 这里是qlen
        attention_scores = attention_scores + key_position_scores_r_t
        return attention_scores


class MllamaTextCrossAttention(MultiHeadAttention):
    '''mllama部分层使用的crossattention'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_norm = LayerNorm(self.attention_head_size, norm_mode='rmsnorm', eps=kwargs.get('layer_norm_eps', 1e-6), bias=False)
        self.k_norm = LayerNorm(self.attention_key_size, norm_mode='rmsnorm', eps=kwargs.get('layer_norm_eps', 1e-6), bias=False)

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        query_states = self.q_norm(query_states)
        bsz = query_states.shape[0]

        if past_key_value is not None:
            key_states, value_states = past_key_value
            attention_mask = encoder_attention_mask
        elif encoder_hidden_states is not None:
            key_states = self.k(encoder_hidden_states)
            value_states = self.v(encoder_hidden_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.attention_key_size).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.attention_key_size).transpose(1, 2)
            key_states = self.k_norm(key_states)
            attention_mask = encoder_attention_mask
        return query_states, key_states, value_states, attention_mask


class ModernBertAttention(RopeAttention):
    def init_position_encoding(self, **kwargs):
        if self.layer_idx % kwargs['global_attn_every_n_layers'] != 0:
            self.local_attention = (kwargs['local_attention'] // 2, kwargs['local_attention'] // 2)
        else:
            self.local_attention = (-1, -1)

        kwargs['rope_theta'] = kwargs['global_rope_theta']
        self.max_position = kwargs['max_position_embeddings']
        if self.local_attention != (-1, -1):
            if kwargs['local_rope_theta'] is not None:
                kwargs['rope_theta'] = kwargs['local_rope_theta']
            self.max_position = kwargs['local_attention']
        super().init_position_encoding(**kwargs)

    def forward(self, 
                hidden_states:Optional[torch.Tensor]=None, 
                attention_mask:Optional[torch.FloatTensor]=None, 
                encoder_hidden_states:Optional[torch.FloatTensor]=None, 
                encoder_attention_mask:Optional[torch.FloatTensor]=None, 
                past_key_value:Optional[Tuple[Tuple[torch.FloatTensor]]]=None, 
                position_ids=None, 
                sliding_window_mask=None,
                **model_kwargs
        ):
        if self.local_attention != (-1, -1):
            attention_mask = sliding_window_mask
        return super().forward(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, 
                               past_key_value=past_key_value, position_ids=position_ids)


ATTENTION_MAP = {
    'MultiHeadAttention': MultiHeadAttention,
    'GatedAttention': GatedAttention,
    'TransformerxlMultiHeadAttn': TransformerxlMultiHeadAttn,
    'DeepseekV2Attention': DeepseekV2Attention,
    'DebertaV2Attention': DebertaV2Attention,
    'AlibiAttention': AlibiAttention,
    'NezhaTypicalRelativeAttention': NezhaTypicalRelativeAttention,
    'RopeAttention': RopeAttention,
    'Qwen3Attention': Qwen3Attention,
    'T5Attention': T5Attention,
    'MllamaTextCrossAttention': MllamaTextCrossAttention,
    'ModernBertAttention': ModernBertAttention,

    # 下面是以p_bias为key
    'deberta_v2': DebertaV2Attention,
    'alibi': AlibiAttention,
    'typical_relative': NezhaTypicalRelativeAttention,
    'rotary': RopeAttention,
    't5_relative': T5Attention,
}