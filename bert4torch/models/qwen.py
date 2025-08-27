from bert4torch.models.transformer import Decoder
from bert4torch.layers.core import Qwen3MoeSparseFeedForward
import torch
import re


class Qwen2(Decoder):
    '''通义千问: https://github.com/QwenLM/Qwen3
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkv有bias, 其余均没有bias
    3) 和InternLM基本一致, 唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True,
                       'mlp_type': 'LlamaFeedForward'})
        super().__init__(*args, **kwargs)
        self.model_type = 'qwen2'
        del self.embeddings.layerNorm

        # 修改网络结构
        for layer in self.decoderLayer:
            layer.feedForward.intermediateDense.register_parameter('bias', None)
            layer.feedForward.outputDense.register_parameter('bias', None)
            layer.feedForward.intermediateDense2.register_parameter('bias', None)
            layer.attnLayerNorm.register_parameter('bias', None)
            layer.ffnLayerNorm.register_parameter('bias', None)
            layer.multiHeadAttention.o.register_parameter('bias', None)
        self.LayerNormFinal.register_parameter('bias', None)

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


class Qwen(Qwen2):
    '''通义千问: https://github.com/QwenLM/Qwen
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkv有bias, 其余均没有bias
    3) 和InternLM基本一致, 唯一的差别是InternLM的multiHeadAttention.o有bias
    4) Qwen2的qkv是分开的，Qwen的qkv是合在一起的
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'qwen'

    def load_trans_ckpt(self, checkpoint):
        '''原始权重中qkv是一个全连接, 需要拆分成q,k,v'''
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            mapping = {
                'transformer.h.%s.attn.c_attn.weight' % i: 'decoderLayer.{}.multiHeadAttention.{}.weight',
                'transformer.h.%s.attn.c_attn.bias' % i: 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                qkv = torch.chunk(qkv, 3, dim=0)
                for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': 'transformer.h.%s.attn.c_attn.weight' % i,
                'decoderLayer.{}.multiHeadAttention.{}.bias': 'transformer.h.%s.attn.c_attn.bias' % i
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    qkv.append(state_dict.pop(model_key.format(i, i_k)))
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典, 格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'transformer.wte.weight',
            'lm_head.weight': 'lm_head.weight',
            'LayerNormFinal.weight': 'transformer.ln_f.weight'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.weight': 'transformer.h.%s.attn.c_proj.weight' % i,
            f'decoderLayer.{i}.attnLayerNorm.weight': 'transformer.h.%s.ln_1.weight' % i,
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': 'transformer.h.%s.mlp.w2.weight' % i,
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': 'transformer.h.%s.mlp.w1.weight' % i,
            f'decoderLayer.{i}.feedForward.outputDense.weight': 'transformer.h.%s.mlp.c_proj.weight' % i,
            f'decoderLayer.{i}.ffnLayerNorm.weight': 'transformer.h.%s.ln_2.weight' % i
            })
        return mapping


class Qwen3(Qwen2):
    '''通义千问: https://github.com/QwenLM/Qwen3
    1) 没有bias, 和llama一致
    2) q和k有q_norm和k_norm
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True,
                       'mlp_type': 'LlamaFeedForward', 'attn_type': 'Qwen3Attention'})
        Decoder.__init__(self, *args, **kwargs)
        self.model_type = 'qwen3'
        del self.embeddings.layerNorm
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        for i in range(self.num_hidden_layers):
            mapping.update({
                f'decoderLayer.{i}.multiHeadAttention.q_norm.weight': f'model.layers.{i}.self_attn.q_norm.weight',
                f'decoderLayer.{i}.multiHeadAttention.k_norm.weight': f'model.layers.{i}.self_attn.k_norm.weight',
            })
        return mapping


class Qwen3Moe(Qwen3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'qwen3_moe'
        self.mlp_only_layers = kwargs.get('mlp_only_layers')
        self.num_experts = kwargs.get('num_experts')
        self.decoder_sparse_step = kwargs.get('decoder_sparse_step')
        self.num_experts = kwargs.get('num_experts')
        for layer_idx, layer in enumerate(self.decoderLayer):
            if (layer_idx not in self.mlp_only_layers) and (
                self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0
            ):
                layer.feedForward = Qwen3MoeSparseFeedForward(**kwargs)
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        mapping = {k:v for k,v in mapping.items() if not re.search('decoderLayer\\.[0-9]+\\.feedForward', k)}

        for i in range(self.num_hidden_layers):
            if (i not in self.mlp_only_layers) and (
                self.num_experts > 0 and (i + 1) % self.decoder_sparse_step == 0
            ):
                mapping[f'decoderLayer.{i}.feedForward.gate.weight'] = f"model.layers.{i}.mlp.gate.weight"
                for j in range(self.num_experts):
                    mapping.update({
                    f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense.weight': f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight',
                    f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense2.weight': f'model.layers.{i}.mlp.experts.{j}.up_proj.weight',
                    f'decoderLayer.{i}.feedForward.experts.{j}.outputDense.weight': f'model.layers.{i}.mlp.experts.{j}.down_proj.weight',
                    })
        return mapping