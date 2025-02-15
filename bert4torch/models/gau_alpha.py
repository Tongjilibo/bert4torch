from bert4torch.models.roformer import RoFormerV2
from torch import nn
import copy
from bert4torch.layers import BlockIdentity, GAULayer


class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm', 'normalization': 'softmax_plus'})
        super().__init__(*args, **kwargs)

        layer = GAULayer(**kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.model_type = 'gau_alpha'

    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        '''在convert脚本里已经把key转成bert4torch可用的
        '''
        return {k: k for k, _ in self.named_parameters()}