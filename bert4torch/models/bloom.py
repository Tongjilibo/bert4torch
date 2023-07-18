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
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'is_decoder': True})
        super().__init__(*args, **kwargs)
    
    def variable_mapping(self, prefix='bloom'):
        return super().variable_mapping(prefix)