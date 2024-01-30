from bert4torch.layers import DeepseekMoE
from bert4torch.models.transformer import Decoder
from bert4torch.layers import LlamaFeedForward


class DeepSeek(Decoder):
    '''DeepSeek: https://github.com/deepseek-ai/DeepSeek-MoE
    模型结构: 基本和llama基本一致, 只是各个linear层多了bias; 和Qwen基本一致, 除了o有bias
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkvo有bias, 其余均没有bias
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        self.model_type = 'deepseek'
        del self.embeddings.layerNorm

        # 修改网络结构
        kwargs.pop('bias')
        self.n_routed_experts = kwargs.get('n_routed_experts')
        self.first_k_dense_replace = kwargs.get('first_k_dense_replace')
        self.moe_layer_freq = kwargs.get('moe_layer_freq')
        for layer_idx, layer in enumerate(self.decoderLayer):
            if layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0:
                layer.feedForward = DeepseekMoE(**kwargs)
            else:
                layer.feedForward = LlamaFeedForward(self.hidden_size, **kwargs)

        self.LayerNormFinal.register_parameter('bias', None)

    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'lm_head.weight': 'lm_head.weight',
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
