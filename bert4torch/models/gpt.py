from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import BertLayer, BlockIdentity
from torch import nn
import torch


class GPT(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 有segment, embedding是token、position、segment三者embedding之和
    2. embedding没有加LayerNormalization层
    """
    def __init__(self, *args, **kwargs):
        kwargs['tie_word_embeddings'] = kwargs.get('tie_word_embeddings', True)
        super(GPT, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        self.model_type = 'gpt'

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        # CDial-GPT的[CLS]是0、[PAD]是1，不符合一般习惯，所以交换一下
        w = state_dict.pop('transformer.tokens_embed.weight')
        state_dict['embeddings.word_embeddings.weight'] = torch.cat([w[1:2], w[:1], w[2:]], axis=0)

        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'transformer.h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is not None:
                    is_weight = ckpt_key.endswith('weight')
                    qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                    for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                        # 对weight转置
                        state_dict[model_key.format(i, i_k)] = i_v.T if is_weight else i_v
                    state_dict.pop(ckpt_key)
            
            mapping = {
                f'transformer.h.{i}.attn.c_proj.weight': f'decoderLayer.{i}.multiHeadAttention.o.weight',  # hdsz-hdsz的全连接
                f'transformer.h.{i}.mlp.c_fc.weight': f'decoderLayer.{i}.feedForward.intermediateDense.weight',  # feed forward 第一层
                f'transformer.h.{i}.mlp.c_proj.weight': f'decoderLayer.{i}.feedForward.outputDense.weight'  # feed forward 第二层
            }
            for ckpt_key, model_key in mapping.items():
                if state_dict.get(ckpt_key) is not None:
                    state_dict[model_key] = state_dict.pop(ckpt_key).T
        return state_dict

    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        w = state_dict.pop('embeddings.word_embeddings.weight')
        state_dict['transformer.tokens_embed.weight'] = torch.cat([w[1:2], w[:1], w[2:]], axis=0)

        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.h.{i}.attn.c_attn.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.h.{i}.attn.c_attn.bias'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    weight_bias = state_dict.pop(model_key.format(i, i_k))
                    qkv.append(weight_bias.T if model_key.endswith('weight') else weight_bias)
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv, dim=1) if model_key.endswith('weight') else torch.cat(qkv)
            
            mapping = {
                f'decoderLayer.{i}.multiHeadAttention.o.weight': f'transformer.h.{i}.attn.c_proj.weight',  # hdsz-hdsz的全连接
                f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'transformer.h.{i}.mlp.c_fc.weight',  # feed forward 第一层
                f'decoderLayer.{i}.feedForward.outputDense.weight': f'transformer.h.{i}.mlp.c_proj.weight'  # feed forward 第二层
            }
            for model_key, ckpt_key in mapping.items():
                if state_dict.get(model_key) is not None:
                    state_dict[ckpt_key] = state_dict.pop(model_key).T

        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.position_embeddings.weight': 'transformer.positions_embed.weight',
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'transformer.h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'transformer.h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'transformer.h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'transformer.h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'transformer.h.{i}.ln_2.bias'
            })
        return mapping


class GPT2(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 没有segment输入
    2. embedding之后没有LayerNormalization层
    3. 使用pre_layernorm, bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
    4. 有final_layernorm
    """
    def __init__(self, *args, **kwargs):
        kwargs['tie_word_embeddings'] = kwargs.get('tie_word_embeddings', True)
        kwargs['pre_layernorm'] = kwargs.get('pre_layernorm', True)
        kwargs['final_layernorm'] = kwargs.get('final_layernorm', True)
        super(GPT2, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        self.model_type = 'gpt2'

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'transformer.h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is not None:
                    is_weight = ckpt_key.endswith('weight')
                    qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                    for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                        state_dict[model_key.format(i, i_k)] = i_v.T if is_weight else i_v
                    state_dict.pop(ckpt_key)

            mapping = {
                f'transformer.h.{i}.attn.c_proj.weight': f'decoderLayer.{i}.multiHeadAttention.o.weight',  # hdsz-hdsz的全连接
                f'transformer.h.{i}.mlp.c_fc.weight': f'decoderLayer.{i}.feedForward.intermediateDense.weight',  # feed forward 第一层
                f'transformer.h.{i}.mlp.c_proj.weight': f'decoderLayer.{i}.feedForward.outputDense.weight'  # feed forward 第二层
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
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.h.{i}.attn.c_attn.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.h.{i}.attn.c_attn.bias'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    weight_bias = state_dict.pop(model_key.format(i, i_k))
                    qkv.append(weight_bias.T if model_key.endswith('weight') else weight_bias)
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv, dim=1) if model_key.endswith('weight') else torch.cat(qkv)
            
            mapping = {
                f'decoderLayer.{i}.multiHeadAttention.o.weight': f'transformer.h.{i}.attn.c_proj.weight',  # hdsz-hdsz的全连接
                f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'transformer.h.{i}.mlp.c_fc.weight',  # feed forward 第一层
                f'decoderLayer.{i}.feedForward.outputDense.weight': f'transformer.h.{i}.mlp.c_proj.weight'  # feed forward 第二层
            }
            for model_key, ckpt_key in mapping.items():
                if state_dict.get(model_key) is not None:
                    state_dict[ckpt_key] = state_dict.pop(model_key).T

        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'transformer.wte.weight',
            'embeddings.position_embeddings.weight': 'transformer.wpe.weight',
            'LayerNormFinal.weight': 'transformer.ln_f.weight',
            'LayerNormFinal.bias': 'transformer.ln_f.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'transformer.h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'transformer.h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'transformer.h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'transformer.h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'transformer.h.{i}.ln_2.bias'
            })
        return mapping


class GPT2_ML(Decoder):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    1. embedding后也有layernorm
    2. 第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    _no_split_modules = ["Gpt2MlLayer"]
    def __init__(self, *args, **kwargs):
        kwargs['tie_word_embeddings'] = kwargs.get('tie_word_embeddings', True)
        kwargs['layer_type'] = "Gpt2MlLayer"
        super().__init__(*args, **kwargs)
        self.model_type = 'gpt2_ml'
    
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
