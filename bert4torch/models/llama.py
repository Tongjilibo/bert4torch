from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments, modify_variable_mapping
from bert4torch.layers import NormHead
import torch
import re


class LLaMA(Decoder):
    '''LLaMA
    链接: https://github.com/facebookresearch/llama
    1. 去掉bias
    2. rmsnorm
    3. feedForward不同, 三层全连接
    4. rotary相对位置编码
    '''
    def __init__(self, *args, p_bias='rotary', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 
                       'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True, 
                       'mlp_type': 'LlamaFeedForward'})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        self.model_type = 'llama'

        # 修改lm_head，目前在Baichuan2中使用
        if kwargs.get('norm_head') is True:
            self.lm_head = NormHead(self.hidden_size, self.vocab_size)
    
    def variable_mapping(self):
        '''映射到权重格式
        llama一般有两种格式, 一种是huggingface格式, 一种是pth格式, 这里的映射是以hf格式为准
        '''
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


class Baichuan(LLaMA):
    '''Baichuan
    单独拎出来是因为qkv是合并的权重W_pack
    '''
    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        # baichuan的qkv权重是合在一起的W_pack, 单独处理
        for i in range(self.num_hidden_layers):
            mapping = {f'model.layers.{i}.self_attn.W_pack.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight'}
            for ckpt_key, model_key in mapping.items():
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                qkv = torch.split(qkv, [self.hidden_size, self.hidden_size, self.hidden_size], 0)
                for i_k, i_v in zip(['q','k', 'v'], qkv):
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {'decoderLayer.{}.multiHeadAttention.{}.weight': f'model.layers.{i}.self_attn.W_pack.weight'}
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if model_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(model_key.format(i, i_k)))
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv)
        return state_dict
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        return {k:v for k, v in mapping.items() if not re.search('(q|k|v)_proj.weight', v)}


class MiniCPM(LLaMA):
    _no_split_modules = ["MiniCPMLayer"]
    def __init__(self, *args, **kwargs):
        kwargs['layer_type'] = 'MiniCPMLayer'
        super().__init__(*args, **kwargs)
        self.logit_scale = 1 / (self.hidden_size / kwargs.get('dim_model_base'))

    @property
    def _layer_args(self):
        args = super()._layer_args
        return args + ['num_hidden_layers']