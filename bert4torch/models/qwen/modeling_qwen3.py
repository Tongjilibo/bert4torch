from bert4torch.models.base import Decoder, register_model
from .modeling_qwen2 import Qwen2


@register_model(name="qwen3")
class Qwen3(Qwen2):
    '''通义千问: https://github.com/QwenLM/Qwen3
    1) 没有bias, 和llama一致
    2) q和k有q_norm和k_norm
    '''
    def __init__(self, *args, **kwargs):
        Decoder.__init__(self, *args, **kwargs)
        del self.embeddings.layerNorm
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        for i in range(self.num_hidden_layers):
            mapping.update({
                f'decoderLayer.{i}.multiHeadAttention.q_norm.weight': f'model.layers.{i}.self_attn.q_norm.weight',
                f'decoderLayer.{i}.multiHeadAttention.k_norm.weight': f'model.layers.{i}.self_attn.k_norm.weight',
            })
        return mapping