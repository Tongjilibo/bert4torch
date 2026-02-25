from bert4torch.models.base import Decoder, register_model


@register_model(name="ernie4_5")
class Ernied4_5(Decoder):
    '''百度Ernied4_5大模型
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
            f'decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping