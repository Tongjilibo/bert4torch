from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import BertLayer, BlockIdentity
from torch import nn
import copy


class GPT(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 有segment, embedding是token、position、segment三者embedding之和
    2. embedding没有加LayerNormalization层
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(GPT, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm

    def load_variable(self, state_dict, name, prefix='gpt'):
        return super(GPT, self).load_variable(state_dict, name, prefix=prefix)

    def variable_mapping(self, prefix='gpt'):
        # 映射到GPT权重格式
        return super(GPT, self).variable_mapping(prefix=prefix)


class GPT2(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 没有segment输入
    2. embedding之后没有LayerNormalization层
    3. 使用pre_layernorm, bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
    4. 有final_layernorm
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        kwargs['pre_layernorm'] = kwargs.get('pre_layernorm', True)
        kwargs['final_layernorm'] = kwargs.get('final_layernorm', True)
        super(GPT2, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm

    def load_variable(self, state_dict, name, prefix='gpt2'):
        return super(GPT2, self).load_variable(state_dict, name, prefix=prefix)

    def variable_mapping(self, prefix='gpt2'):
        # 映射到GPT权重格式
        return super(GPT2, self).variable_mapping(prefix=prefix)


class GPT2_ML(Decoder):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    1. embedding后也有layernorm
    2. 第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super().__init__(*args, **kwargs)
        layer = self.Gpt2MlLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, 
                                 is_dropout=self.is_dropout, conditional_size=self.conditional_size, is_decoder=True)
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix='gpt2_ml')

    def variable_mapping(self):
        # 映射到GPT2权重格式
        return super(GPT2_ML, self).variable_mapping(prefix='gpt2_ml')

    class Gpt2MlLayer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        '''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, use_states=False, **model_kwargs):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
            hidden_states = hidden_states + self.dropout1(self_attn_output[0])
            x = self.layerNorm1(hidden_states, conditional_emb)

            ffn_output = self.feedForward(x)
            # bert的第二个跳跃连接的输入1是经过了multiHeadAttention+layerNorm1的hidden_states, 即这里的x
            # gpt2_ml的第二个跳跃连接的输入1是经过了multiHeadAttention的hidden_states, 不加layerNorm1
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2(hidden_states, conditional_emb)
            if self.is_decoder and use_states:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs
