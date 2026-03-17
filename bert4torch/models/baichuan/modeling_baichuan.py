from ..base import register_model
from ..llama.modeling_llama import LLaMA
import torch
import re


@register_model(name="baichuan")
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
