from ..base import Decoder, register_model
import torch


@register_model(name="gpt2_ml")
class GPT2_ML(Decoder):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    1. embedding后也有layernorm
    2. 第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    _no_split_modules = ["Gpt2MlLayer"]

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                is_weight = ckpt_key.endswith('weight')
                qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                    state_dict[model_key.format(i, i_k)] = i_v.T if is_weight else i_v
                state_dict.pop(ckpt_key)
            
            mapping = {
                f'h.{i}.attn.c_proj.weight': f'decoderLayer.{i}.multiHeadAttention.o.weight',  # hdsz-hdsz的全连接
                f'h.{i}.mlp.c_fc.weight': f'decoderLayer.{i}.feedForward.intermediateDense.weight',  # feed forward 第一层
                f'h.{i}.mlp.c_proj.weight': f'decoderLayer.{i}.feedForward.outputDense.weight'  # feed forward 第二层
            }
            for ckpt_key, model_key in mapping.items():
                if state_dict.get(ckpt_key) is not None:
                    state_dict[model_key] = state_dict.pop(ckpt_key).T

        return state_dict

    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'h.{i}.attn.c_attn.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'h.{i}.attn.c_attn.bias'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    weight_bias = state_dict.pop(model_key.format(i, i_k))
                    qkv.append(weight_bias.T if model_key.endswith('weight') else weight_bias)
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv, dim=1) if model_key.endswith('weight') else torch.cat(qkv)
            
            mapping = {
                f'transformer.h.{i}.attn.c_proj.weight': f'decoderLayer.{i}.multiHeadAttention.o.weight',  # hdsz-hdsz的全连接
                f'transformer.h.{i}.mlp.c_fc.weight': f'decoderLayer.{i}.feedForward.intermediateDense.weight',  # feed forward 第一层
                f'transformer.h.{i}.mlp.c_proj.weight': f'decoderLayer.{i}.feedForward.outputDense.weight'  # feed forward 第二层
            }
            for model_key, ckpt_key in mapping.items():
                if state_dict.get(model_key) is not None:
                    state_dict[ckpt_key] = state_dict.pop(model_key).T

        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'wte.weight',
            'embeddings.position_embeddings.weight': 'wpe.weight',
            'embeddings.layerNorm.weight': 'emb_norm.weight',
            'embeddings.layerNorm.bias': 'emb_norm.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'h.{i}.ln_2.bias'
            })
        return mapping
