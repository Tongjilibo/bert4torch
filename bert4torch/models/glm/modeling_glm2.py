from bert4torch.layers import GlmRMSNorm
from ..base import register_model
from .modeling_glm import Glm
import torch


@register_model(name="chatglm2")
@register_model(name="glm2")
@register_model(name="glm3")  # 架构一样
class Glm2(Glm):
    """CHATGLM2-6B: https://github.com/THUDM/ChatGLM2-6B
    主要修改：1) 不使用Unilm式的mask
             2) flash_attention
             3) multi_query_attention
    """
    _no_split_modules = ["Glm2Layer"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LayerNormFinal = GlmRMSNorm(self.hidden_size, layer_norm_eps=kwargs.get('layer_norm_eps', 1e-5))

    def load_trans_ckpt(self, checkpoint, prefix=''):
        state_dict = super().load_trans_ckpt(checkpoint)
        # weight bias
        for i in range(self.num_hidden_layers):
            mapping = {
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight': prefix + 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias': prefix + 'decoderLayer.{}.multiHeadAttention.{}.bias',
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight_scale': prefix + 'decoderLayer.{}.multiHeadAttention.{}.weight_scale'
            }
            for ckpt_key, model_key in mapping.items():
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                inner_dim = (qkv.shape[0]-self.hidden_size) // 2
                q, k, v = torch.split(qkv, [self.hidden_size, inner_dim, inner_dim], 0)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict

    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias',
                'decoderLayer.{}.multiHeadAttention.{}.weight_scale': f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight_scale'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if model_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(model_key.format(i, i_k)))
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv)
        return state_dict

    def variable_mapping(self, prefix='transformer.encoder'):
        mapping = super().variable_mapping(prefix)
        mapping.update({
            'embeddings.word_embeddings.weight': 'transformer.embedding.word_embeddings.weight',
            'lm_head.weight': "transformer.output_layer.weight"
        })
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layers.%d.' % i
            mapping.update({
                f'decoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'self_attention.dense.weight',
                f'decoderLayer.{i}.multiHeadAttention.o.weight_scale': prefix_i + "self_attention.dense.weight_scale",
                f'decoderLayer.{i}.feedForward.intermediateDense.weight_scale': prefix_i + "mlp.dense_h_to_4h.weight_scale",
                f'decoderLayer.{i}.feedForward.outputDense.weight_scale': prefix_i + "mlp.dense_4h_to_h.weight_scale",
                f'decoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + "self_attention.dense.weight",
            })
        return mapping

    def prepare_inputs(self, *inputs, **model_kwargs):
        return model_kwargs
