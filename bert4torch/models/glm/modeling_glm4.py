from bert4torch.models.base import Decoder, register_model
from bert4torch.models.modeling_utils import safe_register_parameter


@register_model(name="glm4")
class Glm4(Decoder):
    '''
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

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
            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'model.layers.{i}.self_attn.k_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'model.layers.{i}.self_attn.v_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.self_attn.o_proj.weight',
            f'decoderLayer.{i}.input_layernorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.post_attention_layernorm.weight': f'model.layers.{i}.post_self_attn_layernorm.weight',
            f'decoderLayer.{i}.pre_mlp_layernorm.weight': f'model.layers.{i}.post_attention_layernorm.weight',
            f'decoderLayer.{i}.post_mlp_layernorm.weight': f'model.layers.{i}.post_mlp_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping