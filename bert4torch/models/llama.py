from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import BlockIdentity
from bert4torch.activations import get_activation
from torch import nn


class LLaMA(Decoder):
    '''LLaMA
    链接: https://github.com/facebookresearch/llama
    1. 去掉bias
    2. rmsnorm
    3. feedForward不同, 三层全连接
    4. rotary相对位置编码
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # 修改feedword
        for layer in self.decoderLayer:
            layer.feedForward = self.FeedForward(self.hidden_size, **kwargs)
            layer.dropout1 = BlockIdentity()  # llama未使用dropout
            layer.dropout2 = BlockIdentity()

    def load_variable(self, state_dict, name):
        return super(LLaMA, self).load_variable(state_dict, name, prefix='llama')

    def variable_mapping(self, prefix='llama'):
        # 映射到权重格式
        mapping = super(LLaMA, self).variable_mapping(prefix=prefix)
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'decoderLayer.{i}.feedForward.intermediateDense2.weight': prefix_i + 'intermediate2.dense.weight'})
        return mapping
    
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
