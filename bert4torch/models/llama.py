from bert4torch.models.base import LM_Mask
from bert4torch.models.bert import BERT
from bert4torch.snippets import delete_arguments
from bert4torch.layers import LayerNorm, BertLayer, BlockIdentity
from bert4torch.activations import get_activation
from torch import nn
import torch.nn.functional as F
import copy


class LLaMA(LM_Mask, BERT):
    '''LLaMA
    链接: https://github.com/facebookresearch/llama
    改动：模型结构和gpt2类似，去掉bias，简化Norm, feedForward不同
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 'is_decoder': True})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = self.TransformerBlock(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                                    'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, norm_mode=kwargs['norm_mode'], bias=kwargs['bias'])
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False) 
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))
        # 修改feedword
        for layer in self.encoderLayer:
            layer.feedForward = self.FeedForward(self.hidden_size, **kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.tie_emb_prj_weight:
            self.dense.weight = self.embeddings.word_embeddings.weight

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(self.LayerNormFinal(hidden_state))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(LLaMA, self).load_variable(state_dict, name, prefix='llama')

    def variable_mapping(self, prefix='llama'):
        # 映射到权重格式
        mapping = super(LLaMA, self).variable_mapping(prefix=prefix)
        mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight',
                        'dense.weight': f'{prefix}.dense.weight'})
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.feedForward.intermediateDense2.weight': prefix_i + 'intermediate2.dense.weight'})
        return mapping
    
    class TransformerBlock(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            del self.dropout1
            del self.dropout2

        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
            x = self.layerNorm1(hidden_states, conditional_emb)
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
            hidden_states = hidden_states + self_attn_output[0]

            x = self.layerNorm2(hidden_states, conditional_emb)
            hidden_states = hidden_states + self.feedForward(x)
            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs
        
    class FeedForward(nn.Module):
        '''FeedForward和Bert的不一致，Bert只有两个全连接'''
        def __init__(self, dim: int, intermediate_size: int, hidden_act='silu', **kwargs):
            super().__init__()
            self.intermediateDense = nn.Linear(dim, intermediate_size, bias=False)
            self.outputDense = nn.Linear(intermediate_size, dim, bias=False)
            self.intermediateDense2 = nn.Linear(dim, intermediate_size, bias=False)
            self.intermediate_act_fn = get_activation(hidden_act)

        def forward(self, x):
            return self.outputDense(self.intermediate_act_fn(self.intermediateDense(x)) * self.intermediateDense2(x))
