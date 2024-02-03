from torch import nn
import torch
import math
import torch.nn.functional as F
from bert4torch.layers.core import LayerNorm, PositionWiseFeedForward
from bert4torch.layers.attention import MultiHeadAttentionLayer, GatedAttentionUnit


class BertLayer(nn.Module):
    """Transformer层:
       顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

       注意:
        1. 以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
        2. 原始的Transformer的encoder中的Feed Forward层一共有两层linear，
        3. config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, 
                 is_dropout=False, conditional_size=False, pre_layernorm=False, apply_residual_post_layernorm=False, **kwargs):
        super(BertLayer, self).__init__()
        self.dropout_rate = dropout_rate
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.pre_layernorm = pre_layernorm  # True表示pre, False表示post
        self.apply_residual_post_layernorm = apply_residual_post_layernorm
        self.is_decoder = kwargs.get('is_decoder', False)
        self.add_cross_attention = kwargs.get('add_cross_attention', False)
        
        # self attention
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.attnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

        # feedforward
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act, is_dropout=is_dropout, **kwargs)
        self.ffnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

        # cross attention
        if self.add_cross_attention and self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.crossLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

    def forward(self, hidden_states=None, attention_mask=None, position_ids=None, conditional_emb=None, encoder_hidden_states=None, 
                encoder_attention_mask=None, past_key_value=None, cross_past_key_value=None, **model_kwargs):
        return_tensors = dict()
        # ============== self attention ==============
        x = self.attnLayerNorm(hidden_states, conditional_emb) if self.pre_layernorm else hidden_states  # pre/post layernorm
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)  # self.decoder为true时候，这里的attention_mask是三角的
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(self_attn_output[0], residual)
        hidden_states = self.attnLayerNorm(hidden_states, conditional_emb) if not self.pre_layernorm else hidden_states
        
        # ============== cross attention ==============
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.crossLayerNorm(hidden_states, conditional_emb) if self.pre_layernorm else hidden_states  # pre/post layernorm
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value, position_ids=position_ids)
            residual = x if self.apply_residual_post_layernorm else hidden_states
            hidden_states = self.dropout_add(cross_attn_output[0], residual)
            if model_kwargs.get('use_states', False):
                return_tensors['cross_past_key_value'] = cross_attn_output[-1]
            hidden_states = self.crossLayerNorm(hidden_states, conditional_emb) if not self.pre_layernorm else hidden_states

        # ============== feedforward ==============
        x = self.ffnLayerNorm(hidden_states, conditional_emb) if self.pre_layernorm else hidden_states  # pre/post layernorm
        feedforward_output = self.feedForward(x)
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(feedforward_output, residual)
        hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb) if not self.pre_layernorm else hidden_states
        
        if self.is_decoder and model_kwargs.get('use_states', False):
            return_tensors['past_key_value'] = self_attn_output[-1]
        return_tensors['hidden_states'] = hidden_states
        return return_tensors

    def dropout_add(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = F.dropout(x, p=self.dropout_rate, training=self.training)
        out = residual + out
        return out


class T5Layer(BertLayer):
    """T5的Encoder的主体是基于Self-Attention的模块
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """
    def __init__(self, *args, version='t5.1.0', **kwargs):
        super().__init__(*args, **kwargs)

        # 如果是t5.1.1结构，则FFN层需要变更
        if version.endswith('t5.1.1'):
            self.feedForward = self.T5PositionWiseFeedForward(**kwargs)

        # decoder中间有crossAttention
        if self.add_cross_attention and self.is_decoder and hasattr(self.crossAttention, 'relative_positions_encoding'):
            del self.crossAttention.relative_positions_encoding
            del self.crossAttention.relative_positions

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, encoder_hidden_states=None, 
                encoder_attention_mask=None, past_key_value=None, cross_past_key_value=None, **model_kwargs):
        # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)

        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.crossLayerNorm(hidden_states, conditional_emb)
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value)
            hidden_states = self.dropout_add(cross_attn_output[0], hidden_states)
            if model_kwargs.get('use_states', False):
                model_kwargs['cross_past_key_value'] = cross_attn_output[-1]

        # feed forward
        x = self.ffnLayerNorm(hidden_states, conditional_emb)
        ffn_output = self.feedForward(x)
        hidden_states = self.dropout_add(ffn_output, hidden_states)

        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs

    class T5PositionWiseFeedForward(PositionWiseFeedForward):
        '''参考transformer包: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py'''
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
        self.pre_layernorm = kwargs.get('pre_layernorm')
        # multiattn层无bias
        self.multiHeadAttention = self.RelPartialLearnableMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, bias=False, **kwargs)

    def forward(self, hidden_states=None, segment_ids=None, pos_emb=None, attention_mask=None, mems_i=None, conditional_emb=None, **model_kwargs):
        # 拼接mems和query，mems_i: [btz, m_len, hdsz], w: [btz, q_len, hdsz] = [btz, k_len, hdsz]
        hidden_states_cat = torch.cat([mems_i, hidden_states], 1) if mems_i is not None else hidden_states
        
        # Attn
        if self.pre_layernorm:
            hidden_states_cat = self.attnLayerNorm(hidden_states_cat, conditional_emb)
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states_cat, pos_emb, attention_mask, segment_ids)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        if not self.pre_layernorm:  # post_layernorm
            hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)

        # FFN
        x = self.ffnLayerNorm(hidden_states, conditional_emb) if self.pre_layernorm else hidden_states
        self_attn_output2 = self.feedForward(x)
        hidden_states = self.dropout_add(self_attn_output2, hidden_states)
        if not self.pre_layernorm:  # post_layernorm
            hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs

    class RelPartialLearnableMultiHeadAttn(MultiHeadAttentionLayer):
        '''Transformer_XL式相对位置编码, 这里修改成了MultiHeadAttentionLayer的batch_first代码格式'''
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