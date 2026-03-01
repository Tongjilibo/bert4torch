from torch import nn
import torch
import math
import torch.nn.functional as F
from bert4torch.layers.layer_norm import LAYER_NORM, RoformerV2LayerNorm
from bert4torch.layers.mlp import MLP_MAP, T5PositionWiseFeedForward
from bert4torch.layers.attention import ATTENTION_MAP, GatedAttention, TransformerxlMultiHeadAttn
from bert4torch.models.modeling_utils import safe_register_parameter
from bert4torch.snippets import create_registrar
from typing import Union, Optional, Tuple, Dict, Type


TRANSFORMER_BLOCKS : Dict[str, Type[nn.Module]] = {}
register_layer = create_registrar(TRANSFORMER_BLOCKS)


@register_layer
@register_layer(name='default')
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
                 **kwargs
        ):
        super(BertLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.is_decoder = kwargs.get('is_decoder', False)
        self.add_cross_attention = kwargs.get('add_cross_attention', False)
        self.attn_type = kwargs.get('attn_type') or kwargs.get('pos_emb_type') or 'MultiHeadAttention'
        self.mlp_type = kwargs.get('mlp_type', 'PositionWiseFeedForward')
        
        # self attention
        self.multiHeadAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.attnLayerNorm = LAYER_NORM[kwargs](hidden_size, **kwargs)

        # feedforward
        self.feedForward = MLP_MAP[self.mlp_type](hidden_size, intermediate_size, dropout_rate=dropout_rate, 
                                                  hidden_act=hidden_act, is_dropout=is_dropout, **kwargs)
        self.ffnLayerNorm = LAYER_NORM[kwargs](hidden_size, **kwargs)

        # cross attention
        if self.add_cross_attention and self.is_decoder:
            self.crossAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.crossLayerNorm = LAYER_NORM[kwargs](hidden_size, **kwargs)

    def forward(
        self, 
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
        # self.decoder为true时候，这里的attention_mask是三角的
        self_attn_output = self.multiHeadAttention(self._process_before_self_attention(hidden_states, conditional_emb), 
                                                   attention_mask, past_key_value=past_key_value, position_ids=position_ids)
        hidden_states = self._process_after_self_attention(self_attn_output[0], hidden_states, conditional_emb)
        if self.is_decoder and model_kwargs.get('use_states', False):
            return_tensors['past_key_value'] = self_attn_output[-1]

        # ============== cross attention ==============
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output = self.crossAttention(self._process_before_cross_attention(hidden_states, conditional_emb), 
                                                    None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value, position_ids=position_ids)
            hidden_states = self._process_after_cross_attention(cross_attn_output[0], hidden_states, conditional_emb)
            if model_kwargs.get('use_states', False):
                return_tensors['cross_past_key_value'] = cross_attn_output[-1]

        # ============== feedforward/mlp ==============
        mlp_output = self.feedForward(self._process_before_mlp(hidden_states, conditional_emb))
        hidden_states = self._process_after_mlp(mlp_output, hidden_states, conditional_emb)

        return_tensors['hidden_states'] = hidden_states
        return return_tensors

    def dropout_add(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = F.dropout(x, p=self.dropout_rate, training=self.training)
        out = residual + out
        return out

    def _process_before_self_attention(self, hidden_states, conditional_emb):
        '''self attention前处理'''
        return hidden_states
    
    def _process_after_self_attention(self, self_attn_output, hidden_states, conditional_emb):
        '''self attention后处理'''
        hidden_states = self.dropout_add(self_attn_output, hidden_states)
        return self.attnLayerNorm(hidden_states, conditional_emb)
    
    def _process_before_cross_attention(self, hidden_states, conditional_emb):
        '''cross attention前处理'''
        return hidden_states
    
    def _process_after_cross_attention(self, cross_attn_output, hidden_states, conditional_emb):
        '''cross attention后处理'''
        hidden_states = self.dropout_add(cross_attn_output, hidden_states)
        return self.crossLayerNorm(hidden_states, conditional_emb)
    
    def _process_before_mlp(self, hidden_states, conditional_emb):
        '''mlp前处理'''
        return hidden_states
    
    def _process_after_mlp(self, mlp_output, hidden_states, conditional_emb):
        '''mlp后处理'''
        hidden_states = self.dropout_add(mlp_output, hidden_states)
        return self.ffnLayerNorm(hidden_states, conditional_emb)


@register_layer
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


@register_layer
class XlnetLayer(BertLayer):
    '''Transformer_XL层
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    '''
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs):
        super().__init__(hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs)
        # multiattn层无bias
        self.multiHeadAttention = TransformerxlMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, use_bias=False, **kwargs)

    def forward(self, hidden_states=None, segment_ids=None, pos_emb=None, attention_mask=None, mems_i=None, conditional_emb=None, **model_kwargs):
        # 拼接mems和query，mems_i: [btz, m_len, hdsz], w: [btz, q_len, hdsz] = [btz, k_len, hdsz]
        hidden_states_cat = torch.cat([mems_i, hidden_states], 1) if mems_i is not None else hidden_states
        
        # Attn
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states_cat, pos_emb, attention_mask, segment_ids)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)

        # FFN
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = self.dropout_add(self_attn_output2, hidden_states)
        hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs
    

@register_layer
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


@register_layer
class LLMLayer(BertLayer):
    """LLM的Encoder的主体是基于Self-Attention的模块: PreLayerNorm
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """
    
    def _process_before_self_attention(self, hidden_states, conditional_emb):
        '''self attention前处理'''
        return self.attnLayerNorm(hidden_states, conditional_emb)
    
    def _process_after_self_attention(self, self_attn_output, hidden_states, conditional_emb):
        '''self attention后处理'''
        return self.dropout_add(self_attn_output, hidden_states)
    
    def _process_before_cross_attention(self, hidden_states, conditional_emb):
        '''cross attention前处理'''
        return self.crossLayerNorm(hidden_states, conditional_emb)
    
    def _process_after_cross_attention(self, cross_attn_output, hidden_states, conditional_emb):
        '''cross attention后处理'''
        return self.dropout_add(cross_attn_output, hidden_states)
    
    def _process_before_mlp(self, hidden_states, conditional_emb):
        '''mlp前处理'''
        return self.ffnLayerNorm(hidden_states, conditional_emb)
    
    def _process_after_mlp(self, mlp_output, hidden_states, conditional_emb):
        '''mlp后处理'''
        return self.dropout_add(mlp_output, hidden_states)


@register_layer
class MiniCPMLayer(LLMLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_depth = kwargs.get("scale_depth")
        self.num_hidden_layers = kwargs['num_hidden_layers']
    def dropout_add(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + x * (self.scale_depth / math.sqrt(self.num_hidden_layers))


@register_layer
class FalconParallelAttnLayer(LLMLayer):
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


@register_layer
class GlmLayer(LLMLayer):
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


@register_layer
class Glm2Layer(LLMLayer):
    '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        safe_register_parameter([
            self.attnLayerNorm,
            self.multiHeadAttention.o,
            self.feedForward.intermediateDense,
            self.feedForward.outputDense],
            'bias', None
        )


@register_layer
class Glm4Layer(LLMLayer):
    '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        del self.attnLayerNorm
        del self.ffnLayerNorm
        self.input_layernorm = LAYER_NORM[kwargs](hidden_size, **kwargs)
        self.post_attention_layernorm = LAYER_NORM[kwargs](hidden_size, **kwargs)
        self.pre_mlp_layernorm = LAYER_NORM[kwargs](hidden_size, **kwargs)
        self.post_mlp_layernorm = LAYER_NORM[kwargs](hidden_size, **kwargs)

    def _process_before_self_attention(self, hidden_states, conditional_emb):
        return self.input_layernorm(hidden_states, conditional_emb)
    
    def _process_after_self_attention(self, self_attn_output, hidden_states, conditional_emb):
        x= self.post_attention_layernorm(self_attn_output, conditional_emb)
        return self.dropout_add(x, hidden_states)
    
    def _process_before_mlp(self, hidden_states, conditional_emb):
        return self.pre_mlp_layernorm(hidden_states, conditional_emb)

    def _process_after_mlp(self, mlp_output, hidden_states, conditional_emb):
        x = self.post_mlp_layernorm(mlp_output, conditional_emb)
        return self.dropout_add(x, hidden_states)
    

@register_layer
class GauLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gau = GatedAttention(**kwargs)
        self.dropout_rate = kwargs.get('dropout_rate')
        self.attnLayerNorm = RoformerV2LayerNorm(**kwargs)

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, position_ids=None, **model_kwargs):
        gau_hidden_states = self.gau(hidden_states, attention_mask, position_ids)
        hidden_states = hidden_states + F.dropout(gau_hidden_states, p=self.dropout_rate, training=self.training)
        hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs
        

@register_layer
class MllamaCrossAttentionDecoderLayer(LLMLayer):
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
