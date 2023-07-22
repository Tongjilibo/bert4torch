from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments

class Bloom(Decoder):
    '''Bloom: https://arxiv.org/abs/2211.05100
    主要区别就是alibi编码，其他和bert结构一致
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'is_decoder': True, 'final_layernorm': True})
        super().__init__(*args, **kwargs)

    def load_variable(self, state_dict, name, prefix='bloom'):
        return super().load_variable(state_dict, name, prefix=prefix)

    def variable_mapping(self, prefix='bloom'):
        return super().variable_mapping(prefix)