from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import AlibiAttention, BertLayer, BlockIdentity
import math
import torch
from torch import nn


class Falcon(Decoder):
    '''Falcon: https://huggingface.co/tiiuae
    falcon-rw-1b：alibi编码，但是其attention_scale是在+attention_mask后执行的，和bloom、baichuan-13b-chat其他不一样
    falcon-7b/falcon-7b-instruct: rotary, 除了layernorm其他都没有bias，其次使用了multi_query_attn
    '''
    _no_split_modules = ["BertLayer", "FalconParallelAttnLayer"]
    def __init__(self, *args, **kwargs):
        kwargs.update({'weight': True, 'pre_layernorm': True, 'norm_mode': 'torch_buildin', 
                      'is_decoder': True, 'final_layernorm': True, 'attention_scale': False})
        if kwargs.get('p_bias') == 'alibi':
            AlibiAttention.apply_alibi_pos_emb = apply_alibi_pos_emb
        super().__init__(*args, **kwargs)
        self.model_type = 'falcon'
        self.multi_query_attention = kwargs.get('num_key_value_heads') is not None
        del self.embeddings.layerNorm

        if kwargs.get('layer_type') == 'FalconParallelAttnLayer':
            self.LayerNormFinal.bias = nn.Parameter(torch.zeros(kwargs['hidden_size']))

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            mapping = {
                f'transformer.h.{i}.self_attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.h.{i}.self_attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                if not self.multi_query_attention:
                    tensor_list = torch.split(qkv, self.attention_head_size, 0)
                    q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
                    q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
                else:
                    q, k, v = torch.split(qkv, [self.hidden_size, self.attention_head_size, self.attention_head_size], 0)

                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict

    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.h.{i}.self_attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.h.{i}.self_attention.query_key_value.bias'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                if not self.multi_query_attention:
                    for i_k in ['q', 'k', 'v']:
                        if model_key.format(i, i_k) in state_dict:
                            qkv.append(state_dict.pop(model_key.format(i, i_k)).split(self.attention_head_size, 0))
                    if qkv:
                        state_dict[ckpt_key] = torch.cat([torch.cat(i) for i in zip(*qkv)])
                else:
                    for i_k in ['q', 'k', 'v']:
                        if model_key.format(i, i_k) in state_dict:
                            qkv.append(state_dict.pop(model_key.format(i, i_k)))
                    if qkv:
                        state_dict[ckpt_key] = torch.cat(qkv)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'transformer.word_embeddings.weight',
            'lm_head.weight': 'lm_head.weight' if self.with_lm and not self.tie_word_embeddings else 'model.embed_tokens.weight',
            'LayerNormFinal.weight': 'transformer.ln_f.weight',
            'LayerNormFinal.bias': 'transformer.ln_f.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'transformer.h.{i}.self_attention.dense.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'transformer.h.{i}.self_attention.dense.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'transformer.h.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'transformer.h.{i}.input_layernorm.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'transformer.h.{i}.mlp.dense_h_to_4h.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.h.{i}.mlp.dense_h_to_4h.bias',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'transformer.h.{i}.mlp.dense_4h_to_h.weight',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'transformer.h.{i}.mlp.dense_4h_to_h.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'transformer.h.{i}.post_attention_layernorm.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'transformer.h.{i}.post_attention_layernorm.bias'
            })
        return mapping


def apply_alibi_pos_emb(self, attention_scores, key_layer):
    ''' 执行alibi相对位置编码，单独拎出来主要是falcon是在+之后再执行attention_scale的 '''
    input_dtype = attention_scores.dtype
    if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
        attention_scores = attention_scores.to(torch.float32)
        
    key_position_scores_r_t = self.relative_positions_encoding(key_layer)
    attention_scores = attention_scores + key_position_scores_r_t
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    return attention_scores
