from typing import Mapping
from bert4torch.models.base import LM_Mask
from bert4torch.models.bert import BERT
from bert4torch.snippets import delete_arguments
from bert4torch.layers import LayerNorm, BertLayer, BlockIdentity
from bert4torch.activations import get_activation
from torch import nn
import torch.nn.functional as F
import copy


class Bloom(LM_Mask, BERT):
    '''Bloom: https://arxiv.org/abs/2211.05100
    主要区别就是alibi编码，其他和bert结构一致
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'is_decoder': True})
        super().__init__(*args, **kwargs)
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12), conditional_size=self.conditional_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False) 
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))
        self.tie_weights()

    def tie_weights(self):
        if self.tie_emb_prj_weight:
            self.dense.weight = self.embeddings.word_embeddings.weight

    def variable_mapping(self, prefix='bloom'):
        mapping = super().variable_mapping(prefix)
        mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight',
                        'LayerNormFinal.bias': f'{prefix}.LayerNormFinal.bias',
                        'dense.weight': f'{prefix}.dense.weight'})
        return mapping

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(self.LayerNormFinal(hidden_state))
        return self.final_activation(logit)
