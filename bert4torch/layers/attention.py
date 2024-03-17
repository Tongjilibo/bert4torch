from torch import nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from bert4torch.layers.position_encoding import *
from bert4torch.activations import get_activation
from bert4torch.snippets import log_warn_once, is_flash_attn_available, is_xformers_available


if is_xformers_available():
    from xformers import ops as xops


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class MultiHeadAttentionLayer(nn.Module):
    '''多头注意力
    '''
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate=0.1, attention_scale=True,
                 output_attentions=False, bias=True, p_bias=None, use_dynamic_ntk=False, flash_attention=None, use_logn_attn=None, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = kwargs.get('is_decoder', False)
        self.attention_scale = attention_scale
        self.output_attentions = output_attentions
        self.bias = bias
        self.p_bias = p_bias
        self.use_dynamic_ntk = use_dynamic_ntk
        
        # 获取flash_attention的配置项
        self.flash_attention_config = kwargs.get('flash_attention_config', dict())
        self.is_causal = kwargs.get('is_causal', False) or self.flash_attention_config.pop('is_causal', False)
        if ((flash_attention is True) or (flash_attention == 'sdpa')) and (int(torch.__version__.split('.')[0]) < 2):
            log_warn_once('`F.scaled_dot_product_attention` only supported in torch 2.0')
            flash_attention = None
        elif (flash_attention == 'xformers') and (not is_xformers_available()):
            log_warn_once("Xformers is not installed correctly. use `pip install xformers`.")
            flash_attention = None
        elif (flash_attention == 'flash_attn_2') and (not is_flash_attn_available()):
            log_warn_once("flash_attn is not installed correctly. please visit https://github.com/Dao-AILab/flash-attention")
            flash_attention = None
        self.flash_attention = flash_attention
        
        self.use_logn_attn = use_logn_attn # 使用logn_attn
        self.max_position = max_position = kwargs.get('max_position')
        # t5_pegasus_small中hidden_size/num_attention_heads != 0
        # 苏神的roberta small中qk的维度和v不同
        self.attention_head_size = kwargs.get('attention_head_size', int(hidden_size/num_attention_heads))
        self.attention_key_size = kwargs.get('attention_key_size', self.attention_head_size)
        q_inner_dim = self.attention_key_size * num_attention_heads
        k_inner_dim = q_inner_dim
        v_inner_dim = self.attention_head_size * num_attention_heads

        # multi query attention: chatglm中叫multi_query_group_num, llama中叫num_key_value_heads
        if (kwargs.get('multi_query_group_num') is not None) or (kwargs.get('num_key_value_heads') is not None):
            self.multi_query_group_num = kwargs.get('multi_query_group_num') or kwargs.get('num_key_value_heads')
            k_inner_dim_tmp = self.attention_head_size * self.multi_query_group_num
            v_inner_dim_tmp = k_inner_dim_tmp

        # longlora
        if kwargs.get('longlora_group_size') is not None:
            self.longlora_group_size = kwargs.get('longlora_group_size')

        self.q = nn.Linear(hidden_size, q_inner_dim, bias=bias)
        self.k = nn.Linear(hidden_size, k_inner_dim_tmp if hasattr(self, 'multi_query_group_num') else k_inner_dim, bias=bias)
        self.v = nn.Linear(hidden_size, v_inner_dim_tmp if hasattr(self, 'multi_query_group_num') else v_inner_dim, bias=bias)
        self.o = nn.Linear(v_inner_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob) if attention_probs_dropout_prob > 0 else lambda x: x

        if self.p_bias == 'typical_relative':  # nezha
            self.relative_positions_encoding = RelativePositionsEncoding(qlen=max_position, klen=max_position,
                                                                         embedding_size=self.attention_head_size,
                                                                         max_relative_position=kwargs.get('max_relative_position'))
        elif self.p_bias == 'rotary':  # roformer, llama, chatglm
            # position_encoding_2d 目前仅在chatglm中使用
            self.position_encoding_2d = kwargs.get('position_encoding_2d', False)
            embedding_size = self.attention_head_size//2 if self.position_encoding_2d else self.attention_head_size
            self.relative_positions_encoding = RoPEPositionEncoding(embedding_size=embedding_size, 
                                                                    max_seq_len_cache=max_position, 
                                                                    rope_rank=kwargs.get('rope_rank', 'adjacent'), 
                                                                    ntk_alpha=kwargs.get('ntk_alpha', 1.0),
                                                                    rope_ratio=kwargs.get('rope_ratio', 1.0),
                                                                    rope_theta=kwargs.get('rope_theta', 10000.0))
        elif self.p_bias == 't5_relative':  # t5
            self.relative_positions = RelativePositionsEncodingT5(qlen=max_position,  klen=max_position, 
                                                                  relative_attention_num_buckets=kwargs.get('relative_attention_num_buckets'), 
                                                                  is_decoder=kwargs.get('is_decoder'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'), num_attention_heads)
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
            self.relative_positions = RelativePositionsEncodingDebertaV2(qlen=max_position, klen=max_position, 
                                                                         position_buckets=kwargs.get('position_buckets'),
                                                                         max_position=max_position)
            self.relative_positions_encoding = nn.Embedding(max_position, self.hidden_size)
            self.norm_rel_ebd = [x.strip() for x in kwargs.get("norm_rel_ebd", "none").lower().split("|")]
            if "layer_norm" in self.norm_rel_ebd:
                self.layernorm = nn.LayerNorm(self.hidden_size, kwargs.get('layer_norm_eps', 1e-12), elementwise_affine=True)
            self.pos_dropout = nn.Dropout(dropout_rate)
        elif self.p_bias == 'alibi':
            self.relative_positions_encoding = ALiBiPositionsEncoding(num_attention_heads)

    def forward(self, hidden_states=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, position_ids=None, **model_kwargs):
        '''
        hidden_states shape: [batch_size, seq_q, hidden_size]
        attention_mask shape: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        encoder_hidden_states shape: [batch_size, seq_k, hidden_size]
        encoder_attention_mask shape: [batch_size, 1, 1, seq_k]
        past_key_value shape: ([batch_size, num_attention_heads, key_len_cache, attention_head_size], ...)
        '''

        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]
        query_layer = self.transpose_for_q_scores(self.q(hidden_states))
        if self.p_bias == 'rotary':
            # rotary有cache情况下，需要先rope后再和past_key_value concat
            key_layer = self.transpose_for_k_scores(self.k(hidden_states))
            value_layer = self.transpose_for_v_scores(self.v(hidden_states))
            query_layer, key_layer, value_layer = self.apply_rotary_pos_emb(query_layer, key_layer, value_layer, position_ids, past_key_value)
        elif (encoder_hidden_states is not None) and (past_key_value is not None):
            key_layer, value_layer = past_key_value
            attention_mask = encoder_attention_mask
        elif encoder_hidden_states is not None:
            key_layer = self.transpose_for_k_scores(self.k(encoder_hidden_states))
            value_layer = self.transpose_for_v_scores(self.v(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_k_scores(self.k(hidden_states))
            value_layer = self.transpose_for_v_scores(self.v(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_k_scores(self.k(hidden_states))
            value_layer = self.transpose_for_v_scores(self.v(hidden_states))

        # 使用logn_attn
        if self.use_logn_attn:
            query_layer *= ((position_ids + 1)[:, None, :, None].log() / np.log(self.max_position)).clip(1).to(query_layer.dtype)

        # past_key_values
        if self.is_decoder and (not self.training):  # 仅推理是记录
            past_key_value = (key_layer, value_layer)

        # multi_query_attention
        if hasattr(self, 'multi_query_group_num') and self.multi_query_group_num > 1:
            key_layer = self.repeat_kv(key_layer)
            value_layer = self.repeat_kv(value_layer)

        # longlora
        if hasattr(self, 'longlora_group_size'):
            query_layer, key_layer, value_layer, attention_mask = self.longlora_shift(query_layer, key_layer, value_layer, attention_mask)

        # ====================================是否使用flash_attention加速====================================
        # xformers
        attention_scores = None
        if (self.flash_attention == 'xformers') and self.training:
            context_layer = xops.memory_efficient_attention(query_layer, key_layer, value_layer, attn_bias=xops.LowerTriangularMask())
        # SDPA
        elif self.flash_attention in {True, 'sdpa'}:
            # is_causal=True 适用于qlen=klen, 测试下来is_causal=True训练更快
            if self.is_causal and (query_layer.shape[2] == key_layer.shape[2]):
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, is_causal=True)
            elif len(self.flash_attention_config) == 0:
                # 默认方式
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask.bool())
            else:
                # 自定义
                with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                    context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask.bool())
        # flash_attn
        elif self.flash_attention == 'flash_attn_2':
            context_layer = self.flash_attention_forward(query_layer, key_layer, value_layer, attention_mask, hidden_states.shape[1])
        # torch原生实现
        else:
            context_layer, attention_scores = self.torch_attention_forward(query_layer, key_layer, value_layer, attention_mask)
        
        if hasattr(self, 'longlora_group_size'):  # context_layer: [bsz * (q_len // group_size), num_heads, group_size, head_dim]
            bsz, q_len = hidden_states.shape[:2]
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.reshape(bsz, q_len, self.num_attention_heads, self.attention_head_size)
            # shift back
            context_layer[:, :, self.num_attention_heads//2:] = context_layer[:, :, self.num_attention_heads//2:].roll(self.longlora_group_size//2, dims=1)
            context_layer = context_layer.reshape(bsz, q_len, self.hidden_size)
        else:
            # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
            context_layer = context_layer.permute(0, 2, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size()[-2]*context_layer.size()[-1],)
            context_layer = context_layer.reshape(*new_context_layer_shape)

        # 是否返回attention scores
        outputs = (self.o(context_layer), attention_scores) if self.output_attentions else (self.o(context_layer),)
        return outputs + (past_key_value,) if self.is_decoder else outputs
    
    def repeat_kv(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(2)
        hidden_states = hidden_states.expand(-1, -1, self.num_attention_heads // self.multi_query_group_num, -1, -1)
        hidden_states = hidden_states.contiguous().view(hidden_states.shape[:1] + (self.num_attention_heads,) + hidden_states.shape[-2:])
        return hidden_states

    def longlora_shift(self, query_states, key_states, value_states, attention_mask):
        '''longlora中对qkv和mask进行修改: https://github.com/dvlab-research/LongLoRA'''
        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]

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
        if hasattr(self, 'multi_query_group_num'):
            new_x_shape = x.size()[:-1] + (self.multi_query_group_num, self.attention_key_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_key_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_v_scores(self, x):
        if hasattr(self, 'multi_query_group_num'):
            new_x_shape = x.size()[:-1] + (self.multi_query_group_num, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def apply_alibi_pos_emb(self, attention_scores, key_layer):
        ''' 执行alibi相对位置编码，单独拎出来主要是falcon是在+之后再执行attention_scale的 '''
        key_position_scores_r_t = self.relative_positions_encoding(key_layer)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = torch.max(attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min))  # baichuan-13b逻辑
        return attention_scores

    def apply_rotary_pos_emb(self, query_layer, key_layer, value_layer, position_ids, past_key_value):
        ''' 执行rotary相对位置编码 '''
        kv_seq_len = key_layer.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        if self.use_dynamic_ntk:
            # rotary的ntk，其实仅仅在step=1时候会触发
            if kv_seq_len == query_layer.shape[-2]:
                context_value = math.log(kv_seq_len / self.max_position, 2) + 1
                ntk_alpha = 2 ** math.ceil(context_value) - 1
                self.relative_positions_encoding.reset_ntk_alpha(max(ntk_alpha, 1))

        if self.position_encoding_2d:  # chatglm独有逻辑
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            if len(position_ids.shape) == 3:
                q1, k1 = self.relative_positions_encoding([q1, k1], position_ids[:, 0, :], kv_seq_len)
                q2, k2 = self.relative_positions_encoding([q2, k2], position_ids[:, 1, :], kv_seq_len)
            else:
                q1, k1 = self.relative_positions_encoding([q1, k1], position_ids, kv_seq_len)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:  # 原rotary逻辑
            query_layer, key_layer = self.relative_positions_encoding([query_layer, key_layer], position_ids, kv_seq_len)
        
        if past_key_value is not None:  # 过了rope再concat
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        return query_layer, key_layer, value_layer

    def apply_deberta_pos_emb(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        '''deberta_v2使用，和原版区别是query_layer是4维, 原disentangled_attention_bias'''
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
            pos_query_layer = self.transpose_for_q_scores(self.q(rel_embeddings)).repeat(btz, 1, 1, 1)
            pos_key_layer = self.transpose_for_k_scores(self.k(rel_embeddings)).repeat(btz, 1, 1, 1)
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
    
    def torch_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
        '''qkv attention: torch原生实现'''
        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            # ==================== nezha相对位置编码 ====================
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
            # ==================== t5相对位置编码 ====================
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])  # 都是klen, 应对use_state=True
            key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
            key_position_scores_r_t = key_position_scores_r_t[:, :, -attention_scores.shape[-2] :, :]  # 这里是qlen
            attention_scores = attention_scores + key_position_scores_r_t
        elif (self.p_bias == 'deberta_v2') and hasattr(self, 'relative_positions_encoding'):
            # ==================== deberta_v2相对位置编码 ====================
            self.attention_scale = False  # deberta_v2使用自己的attention_scale
            scale_factor = 1
            if "c2p" in self.pos_att_type:
                scale_factor += 1
            if "p2c" in self.pos_att_type:
                scale_factor += 1
            scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
            attention_scores = attention_scores / scale.to(dtype=query_layer.dtype)

            rel_embeddings = self.pos_dropout(self.layernorm(self.relative_positions_encoding.weight))
            relations_keys = self.relative_positions(attention_scores.shape[-2], attention_scores.shape[-1])
            rel_att = self.apply_deberta_pos_emb(query_layer, key_layer, relations_keys, rel_embeddings, scale_factor)
            attention_scores = attention_scores + rel_att

        if self.attention_scale:
            # 是否进行attention scale
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # ==================== alibi相对位置编码 ====================
        if (self.p_bias == 'alibi') and hasattr(self, 'relative_positions_encoding'):
            attention_scores = self.apply_alibi_pos_emb(attention_scores, key_layer)

        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
        if attention_mask is not None:
            # attention_mask = attention_mask * attention_mask.squeeze(-2).unsqueeze(-1)  # deberta_v2中使用，但是不使用也不影响
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)  # 下一行的另一种写法
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_layer.dtype).min  # 原来逻辑是-10000，所以传入的mask的非padding部分为1, padding部分为0
            attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=query_layer.dtype)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_attention_heads, query_len, attention_head_size]

        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            # ==================== nezha相对位置编码 ====================
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

        return context_layer, attention_scores
    
    def flash_attention_forward(self, query_layer, key_layer, value_layer, attention_mask, query_length, softmax_scale=None):
        """ flash_attn，参考transformers中的调用
        """
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):       
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
            batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

            key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
            value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
            if query_length == kv_seq_len:
                query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k)
                cu_seqlens_q = cu_seqlens_k
                max_seqlen_in_batch_q = max_seqlen_in_batch_k
                indices_q = indices_k
            elif query_length == 1:
                max_seqlen_in_batch_q = 1
                cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)  # There is a memcpy here, that is very bad.
                indices_q = cu_seqlens_q[:-1]
                query_layer = query_layer.squeeze(1)
            else:
                # The -q_len: slice assumes left padding.
                attention_mask = attention_mask[:, -query_length:]
                query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

            return (query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k),)

        dropout = 0.0 if not self.training else self.attention_probs_dropout_prob
        query_states = query_layer.transpose(1,2)  # [batch_size, query_len, num_attention_heads, attention_head_size]
        key_states = key_layer.transpose(1,2)
        value_states = value_layer.transpose(1,2)
        
        if (not self.is_causal) and (attention_mask.shape[1:3] == torch.Size([1,1])):
            # flash attention目前仅支持key_padding_mask
            attn_mask = attention_mask[:,0,0,:]  # 将4维的attention_mask降低为2维
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                self, query_states, key_states, value_states, attn_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=False)
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not self.is_causal:
            # 使用torch的attention计算
            log_warn_once( 'Flash Attention only support key_padding_mask, use torch_attention_forward instead.')
            attn_output, _ = self.torch_attention_forward(query_layer, key_layer, value_layer, attention_mask)
            self.flash_attention = None
            return attn_output
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True)

        return attn_output.transpose(1,2)

class GatedAttentionUnit(nn.Module):
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
            self.relative_positions_encoding = RoPEPositionEncoding(embedding_size=self.attention_head_size, **kwargs)

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
