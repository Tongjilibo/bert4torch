from bert4torch.models.base import Decoder, register_model
from bert4torch.models.modeling_utils import safe_register_parameter


@register_model(name="qwen2")
class Qwen2(Decoder):
    '''通义千问: https://github.com/QwenLM/Qwen3
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkv有bias, 其余均没有bias
    3) 和InternLM基本一致, 唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # 修改网络结构
        for layer in self.decoderLayer:
            safe_register_parameter([layer.multiHeadAttention.o], 'bias', None)

    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'lm_head.weight': 'lm_head.weight' if self.with_lm and not self.tie_word_embeddings else 'model.embed_tokens.weight',
            'LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.q.weight': f'model.layers.{i}.self_attn.q_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.q.bias': f'model.layers.{i}.self_attn.q_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'model.layers.{i}.self_attn.k_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.k.bias': f'model.layers.{i}.self_attn.k_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'model.layers.{i}.self_attn.v_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.v.bias': f'model.layers.{i}.self_attn.v_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.self_attn.o_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'model.layers.{i}.self_attn.o_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping