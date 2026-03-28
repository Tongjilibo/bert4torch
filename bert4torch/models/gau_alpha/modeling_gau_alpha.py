from bert4torch.models.roformer import RoFormerV2
from ..base import register_model
from torch import nn
import copy
from bert4torch.layers import BlockIdentity, GauLayer, LayerNorm


@register_model(name="gau_alpha")
class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, layer_type='GauLayer', **kwargs)

    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        '''在convert脚本里已经把key转成bert4torch可用的
        '''
        return {k: k for k, _ in self.named_parameters()}