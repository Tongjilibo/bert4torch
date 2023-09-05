from bert4torch.models.transformer import Decoder
from bert4torch.layers import LlamaFeedForward, BlockIdentity


class InternLM(Decoder):
    '''InternLM: https://github.com/InternLM/InternLM
    模型结构：基本和llama基本一致，只是各个linear层多了bias；和Qwen基本一致，除了o有bias
    1）FeedForward和Llama一致，三个dense层
    2）除了qkvo有bias，其余均没有bias
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        self.name = 'internlm'
        del self.embeddings.layerNorm

        # 修改网络结构
        kwargs.pop('bias')
        for layer in self.decoderLayer:
            layer.feedForward = LlamaFeedForward(self.hidden_size, **kwargs)
            layer.dropout1 = BlockIdentity()  # 未使用dropout
            layer.dropout2 = BlockIdentity()
            layer.layerNorm1.register_parameter('bias', None)
            layer.layerNorm2.register_parameter('bias', None)
        self.LayerNormFinal.register_parameter('bias', None)

    def variable_mapping(self):
        # 映射到权重格式
        mapping = super().variable_mapping()
        for i in range(self.num_hidden_layers):
            prefix_i = f'{self.name}.encoder.layer.%d.' % i
            mapping.update({f'decoderLayer.{i}.feedForward.intermediateDense2.weight': prefix_i + 'intermediate2.dense.weight'})
        return mapping
