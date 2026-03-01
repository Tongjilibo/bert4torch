from bert4torch.models.base import Decoder, register_model
from bert4torch.layers.mlp import DeepseekMoeFeedForward


@register_model(name="deepseek_v2")
class DeepSeekV2(Decoder):
    '''
    DeepSeekV2: https://github.com/deepseek-ai/DeepSeek-V2
    DeepSeek-MoE: https://github.com/deepseek-ai/DeepSeek-MoE
    模型结构: 基本和llama基本一致, 只是各个linear层多了bias; 和Qwen基本一致, 除了o有bias
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkvo有bias, 其余均没有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 修改网络结构
        del self.embeddings.layerNorm
        kwargs.pop('use_bias')
        self.n_routed_experts = kwargs.get('n_routed_experts')
        self.first_k_dense_replace = kwargs.get('first_k_dense_replace')
        self.moe_layer_freq = kwargs.get('moe_layer_freq')
        for layer_idx, layer in enumerate(self.decoderLayer):
            if self.n_routed_experts is not None and layer_idx >= self.first_k_dense_replace \
                and layer_idx % self.moe_layer_freq == 0:
                layer.feedForward = DeepseekMoeFeedForward(**kwargs)

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
                f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
            # attn_type为DeepseekV2Attention时候，网络结构会不同
            if self.attn_type is not None and self.attn_type == 'DeepseekV2Attention':
                mapping.update( 
                {
                    f'decoderLayer.{i}.multiHeadAttention.kv_a_proj_with_mqa.weight': f'model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight',
                    f'decoderLayer.{i}.multiHeadAttention.kv_a_layernorm.weight': f'model.layers.{i}.self_attn.kv_a_layernorm.weight',
                    f'decoderLayer.{i}.multiHeadAttention.kv_b.weight': f'model.layers.{i}.self_attn.kv_b_proj.weight'
                })
                
            if i >= self.first_k_dense_replace and i % self.moe_layer_freq == 0:
                mapping.update(
                    {
                        f'decoderLayer.{i}.feedForward.gate.weight': f"model.layers.{i}.mlp.gate.weight",
                        f'decoderLayer.{i}.feedForward.shared_experts.outputDense.weight': f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                        f'decoderLayer.{i}.feedForward.shared_experts.intermediateDense.weight': f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                        f'decoderLayer.{i}.feedForward.shared_experts.intermediateDense2.weight': f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
                    }
                )
                for j in range(self.n_routed_experts):
                    mapping.update(
                        {
                            f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense.weight': f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight",
                            f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense2.weight': f"model.layers.{i}.mlp.experts.{j}.up_proj.weight",
                            f'decoderLayer.{i}.feedForward.experts.{j}.outputDense.weight': f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"
                        })
            else:
                mapping.update(
                {
                    f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
                    f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
                    f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
                })
        return mapping