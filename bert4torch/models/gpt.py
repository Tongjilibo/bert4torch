from bert4torch.models.base import LM_Mask
from bert4torch.models.bert import BERT
from bert4torch.snippets import delete_arguments
from bert4torch.activations import get_activation
from bert4torch.layers import LayerNorm, BertLayer, BlockIdentity
from torch import nn
import copy


class GPT(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    Todo: 理论上gpt系列应该从Decoder继承，自动实现is_decoder=True，且使用decoderLayer
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        """GPT的embedding是token、position、segment三者embedding之和，跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT, self).__init__(*args, is_decoder=True, **kwargs)
        del self.embeddings.layerNorm
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.lm_head(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT, self).load_variable(state_dict, name, prefix='gpt')

    def variable_mapping(self):
        # 映射到GPT权重格式
        mapping =  super(GPT, self).variable_mapping(prefix='gpt')
        return mapping


class GPT2(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        """GPT2的embedding是token、position两者embedding之和。
           1、跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           2、bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT2, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = self.Gpt2Layer(is_decoder=True, **self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                                              'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=kwargs.get('bias', True))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False) 
        self.lm_head.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.lm_head(self.LayerNormFinal(hidden_state))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2, self).load_variable(state_dict, name, prefix='gpt2')

    def variable_mapping(self):
        # 映射到GPT权重格式
        mapping = super(GPT2, self).variable_mapping(prefix='gpt2')
        mapping.update({'LayerNormFinal.weight': 'gpt2.LayerNormFinal.weight',
                        'LayerNormFinal.bias': 'gpt2.LayerNormFinal.bias'})
        return mapping
    
    class Gpt2Layer(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
            x = self.layerNorm1(hidden_states, conditional_emb)
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value)
            hidden_states = hidden_states + self.dropout1(self_attn_output[0])

            x = self.layerNorm2(hidden_states, conditional_emb)
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)

            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs


class GPT2_ML(LM_Mask, BERT):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    看完ckpt中的key，和GPT的区别是embedding后也有layernorm，和bert的区别是第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layer = self.Gpt2MlLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, 
                                 is_dropout=self.is_dropout, conditional_size=self.conditional_size, is_decoder=True)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.lm_head(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix='gpt2_ml')

    def variable_mapping(self):
        # 映射到GPT2权重格式
        mapping =  super(GPT2_ML, self).variable_mapping(prefix='gpt2_ml')
        return mapping

    class Gpt2MlLayer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        '''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
            hidden_states = hidden_states + self.dropout1(self_attn_output[0])
            x = self.layerNorm1(hidden_states, conditional_emb)

            ffn_output = self.feedForward(x)
            # bert的第二个跳跃连接的输入1是经过了multiHeadAttention+layerNorm1的hidden_states, 即这里的x
            # gpt2_ml的第二个跳跃连接的输入1是经过了multiHeadAttention的hidden_states, 不加layerNorm1
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2(hidden_states, conditional_emb)
            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs
