from bert4torch.models.transformer import Decoder
from bert4torch.layers import LlamaFeedForward, BlockIdentity


class Qwen(Decoder):
    '''通义千问: https://github.com/QwenLM/Qwen-7B
    1）FeedForward和Llama一致，三个dense层
    2）除了qkv有bias，其余均没有bias
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # 修改网络结构
        kwargs.pop('bias')
        for layer in self.decoderLayer:
            layer.feedForward = LlamaFeedForward(self.hidden_size, **kwargs)
            layer.dropout1 = BlockIdentity()  # 未使用dropout
            layer.dropout2 = BlockIdentity()
            layer.multiHeadAttention.o.register_parameter('bias', None)
            layer.layerNorm1.register_parameter('bias', None)
            layer.layerNorm2.register_parameter('bias', None)
        self.LayerNormFinal.register_parameter('bias', None)

    def load_variable(self, state_dict, name, prefix='qwen'):
        return super().load_variable(state_dict, name, prefix=prefix)

    def variable_mapping(self, prefix='qwen'):
        # 映射到权重格式
        mapping = super().variable_mapping(prefix=prefix)
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'decoderLayer.{i}.feedForward.intermediateDense2.weight': prefix_i + 'intermediate2.dense.weight'})
        return mapping
