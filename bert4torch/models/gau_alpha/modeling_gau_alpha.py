from bert4torch.models.roformer import RoFormerV2
from bert4torch.models.base import register_model
from torch import nn
import copy
from bert4torch.layers import BlockIdentity, GAULayer, LayerNorm


@register_model(name="gau_alpha")
class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layer = GAULayer(**kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        # LayerNorm没有weight
        for layer in self.modules():
            if isinstance(layer, LayerNorm) and hasattr(layer, 'weight'):
                del layer.weight

    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        '''在convert脚本里已经把key转成bert4torch可用的
        '''
        return {k: k for k, _ in self.named_parameters()}