from torch import nn
import torch
import math
import torch.nn.functional as F
from bert4torch.layers.core import LayerNorm, MLP_MAP, T5PositionWiseFeedForward
from bert4torch.layers.attention import ATTENTION_MAP, GatedAttention, TransformerxlMultiHeadAttn
from typing import Union, Optional, Tuple


class BertLayer(nn.Module):
    """Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

        :param hidden_size: int, 隐含层神经元个数
        :param num_attention_heads: int, 多头注意力的多头数
        :param attention_probs_dropout_prob: float，softmax后的dropout rate
        :param dropout_rate: float, 残差连接中对multiHeadAttention或者mlp添加dropout的rate
        :param intermediate_size: int, mlp中间隐含层的神经元个数，一般是hidden_size的数倍
        :param hidden_act: str，激活函数的种类
        :param is_dropout: bool, mlp中是否使用dropout层，默认为False
        :param conditional_size: bool/int，LayerNorm时候是否使用条件LayerNorm, 默认为False
        :param pre_layernorm: bool, layernorm是pre还是post，bert是post，现在大模型基本都是pre, 默认为False表示post_layernorm
        :param apply_residual_post_layernorm: bool，残差连接时候是使用layernorm前的还是后的hidden_states, 默认为False表示使用layernorm前的

        注意:
        1. 以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
        2. 原始的Transformer的encoder中的Feed Forward层一共有两层linear，
        3. config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """
    def __init__(self, 
                 hidden_size:int, 
                 num_attention_heads:int, 
                 dropout_rate:float, 
                 attention_probs_dropout_prob:float, 
                 intermediate_size:int, 
                 hidden_act:str, 
                 is_dropout:bool=False, 
                 conditional_size:Union[bool, int]=False, 
                 pre_layernorm:bool=False, 
                 apply_residual_post_layernorm:bool=False, 
                 **kwargs
        ):
        super(BertLayer, self).__init__()
        self.dropout_rate = dropout_rate
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.pre_layernorm = pre_layernorm  # True表示pre, False表示post
        self.apply_residual_post_layernorm = apply_residual_post_layernorm
        self.is_decoder = kwargs.get('is_decoder', False)
        self.add_cross_attention = kwargs.get('add_cross_attention', False)
        self.attn_type = kwargs.get('attn_type',  kwargs.get('p_bias', 'MultiHeadAttention'))
        self.mlp_type = kwargs.get('mlp_type', 'PositionWiseFeedForward')
        
        # self attention
        self.multiHeadAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.attnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

        # feedforward
        kwargs['bias'] = kwargs.get('bias', False)
        self.feedForward = MLP_MAP[self.mlp_type](hidden_size, intermediate_size, dropout_rate=dropout_rate, 
                                                  hidden_act=hidden_act, is_dropout=is_dropout, **kwargs)
        self.ffnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

        # cross attention
        if self.add_cross_attention and self.is_decoder:
            self.crossAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.crossLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

    def forward(self, 
                hidden_states:torch.FloatTensor=None, 
                attention_mask:torch.Tensor=None, 
                position_ids:torch.FloatTensor=None, 
                conditional_emb:Optional[torch.Tensor]=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask:Optional[torch.FloatTensor]=None, 
                past_key_value:Optional[Tuple[Tuple[torch.FloatTensor]]]=None, 
                cross_past_key_value:Optional[Tuple[Tuple[torch.FloatTensor]]]=None, 
                **model_kwargs
        ):
        return_tensors = dict()
        # ============== self attention ==============
        # pre layernorm
        if self.pre_layernorm:
            x = self.attnLayerNorm(hidden_states, conditional_emb)
        else:
            x = hidden_states
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)  # self.decoder为true时候，这里的attention_mask是三角的
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(self_attn_output[0], residual)
        # post layernorm
        if not self.pre_layernorm:
            hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        
        # ============== cross attention ==============
        if self.is_decoder and encoder_hidden_states is not None:
            # pre layernorm
            if self.pre_layernorm:
                x = self.crossLayerNorm(hidden_states, conditional_emb)
            else:
                x = hidden_states
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value, position_ids=position_ids)
            residual = x if self.apply_residual_post_layernorm else hidden_states
            hidden_states = self.dropout_add(cross_attn_output[0], residual)
            if model_kwargs.get('use_states', False):
                return_tensors['cross_past_key_value'] = cross_attn_output[-1]
            # post layernorm
            if not self.pre_layernorm:
                hidden_states = self.crossLayerNorm(hidden_states, conditional_emb)

        # ============== feedforward ==============
        # pre layernorm
        if self.pre_layernorm:
            x = self.ffnLayerNorm(hidden_states, conditional_emb)
        else:
            x = hidden_states
        feedforward_output = self.feedForward(x)
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(feedforward_output, residual)
        if not self.pre_layernorm:
            hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        
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
            self.feedForward = T5PositionWiseFeedForward(**kwargs)

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


class XlnetLayer(BertLayer):
    '''Transformer_XL层
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    '''
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs):
        super().__init__(hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs)
        self.pre_layernorm = kwargs.get('pre_layernorm')
        # multiattn层无bias
        self.multiHeadAttention = TransformerxlMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, bias=False, **kwargs)

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
    

class MiniCPMLayer(BertLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_depth = kwargs.get("scale_depth")
        self.num_hidden_layers = kwargs['num_hidden_layers']
    def dropout_add(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + x * (self.scale_depth / math.sqrt(self.num_hidden_layers))


class FalconParallelAttnLayer(BertLayer):
    '''适用于Falcon的transformer block
    主要区别是attention和feedForward是平行的
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attnLayerNorm.bias = nn.Parameter(torch.zeros(kwargs['hidden_size']))
        del self.ffnLayerNorm

    def forward(self, hidden_states=None, attention_mask=None, position_ids=None, conditional_emb=None, past_key_value=None, **model_kwargs):
        # ============== self attention ==============
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)  # self.decoder为true时候，这里的attention_mask是三角的
        
        # ============== feedforward ==============
        feedforward_output = self.feedForward(x)
        feedforward_output += self_attn_output[0]
        hidden_states = self.dropout_add(feedforward_output, hidden_states)

        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class GlmLayer(BertLayer):
    '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = kwargs['num_hidden_layers']
        hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-5)
        self.attnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.ffnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states=None, attention_mask=None, past_key_value=None, **model_kwargs):
        # 和bert区别有两点, 一个是有alpha, 还有一个是跳跃链接用的是经过了layernorm后的
        x = self.attnLayerNorm(hidden_states)
        alpha = (2 * self.num_hidden_layers) ** 0.5
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
        hidden_states = x * alpha + self_attn_output[0]

        x = self.ffnLayerNorm(hidden_states)
        hidden_states = x *alpha +  self.feedForward(x)

        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class Glm2Layer(BertLayer):
    '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attnLayerNorm.register_parameter('bias', None)
        # self.ffnLayerNorm.register_parameter('bias', None)
        self.multiHeadAttention.o.register_parameter('bias', None)
        self.feedForward.intermediateDense.register_parameter('bias', None)
        self.feedForward.outputDense.register_parameter('bias', None)


class Gpt2MlLayer(BertLayer):
    '''未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
    顺序：Att --> Add --> LN --> FFN --> Add --> LN
    '''
    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
        # attn
        self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        x = self.attnLayerNorm(hidden_states, conditional_emb)

        # ffn
        ffn_output = self.feedForward(x)
        # bert的第二个跳跃连接的输入1是经过了multiHeadAttention+attnLayerNorm的hidden_states, 即这里的x
        # gpt2_ml的第二个跳跃连接的输入1是经过了multiHeadAttention的hidden_states, 不加attnLayerNorm
        hidden_states = self.dropout_add(ffn_output, hidden_states)
        hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)

        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class GAULayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gau = GatedAttention(**kwargs)
        self.dropout_rate = kwargs.get('dropout_rate')
        self.attnLayerNorm = LayerNorm(**kwargs)

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, position_ids=None, **model_kwargs):
        gau_hidden_states = self.gau(hidden_states, attention_mask, position_ids)
        hidden_states = hidden_states + F.dropout(gau_hidden_states, p=self.dropout_rate, training=self.training)
        hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs
        

class MllamaCrossAttentionDecoderLayer(BertLayer):
    '''mllama的cross_attention版本'''
    def __init__(self, *args, **kwargs):
        kwargs['attn_type'] = 'MllamaTextCrossAttention'
        super().__init__(*args, **kwargs)
        self.crossAttention = self.multiHeadAttention  # 重命名
        del self.multiHeadAttention
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, cross_attention_states=None, 
                cross_attention_mask=None, cross_past_key_value=None, **model_kwargs):
        residual = hidden_states
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        cross_attn_output = self.crossAttention(x, attention_mask, cross_attention_states, cross_attention_mask, past_key_value=cross_past_key_value)

        hidden_states = residual + self.cross_attn_attn_gate.tanh() * cross_attn_output[0]

        residual = hidden_states
        hidden_states = self.ffnLayerNorm(hidden_states)
        hidden_states = self.feedForward(hidden_states)
        if model_kwargs.get('full_text_row_masked_out_mask') is not None:
            hidden_states = model_kwargs['full_text_row_masked_out_mask'][:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['cross_past_key_value'] = cross_attn_output[-1]

        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


TRANSFORMER_BLOCKS = {
    "BertLayer": BertLayer,
    "MiniCPMLayer": MiniCPMLayer,
    "FalconParallelAttnLayer": FalconParallelAttnLayer,
    "GlmLayer": GlmLayer,
    "Glm2Layer": Glm2Layer,
    "T5Layer": T5Layer,
    "GAU_Layer": GAULayer,
    "Gpt2MlLayer": Gpt2MlLayer,
    "XlnetLayer": XlnetLayer
}