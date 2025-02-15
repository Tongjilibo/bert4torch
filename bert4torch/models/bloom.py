from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
import torch


class Bloom(Decoder):
    '''Bloom: https://arxiv.org/abs/2211.05100
    主要区别就是alibi编码，其他和bert结构一致
    '''
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'is_decoder': True, 'final_layernorm': True})
        super().__init__(*args, **kwargs)
        self.model_type = 'bloom'

    def load_trans_ckpt(self, checkpoint):
        '''原始权重中qkv是一个全连接, 需要拆分成q,k,v'''
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            mapping = {
                f'h.{i}.self_attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'h.{i}.self_attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for ckpt_key, model_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                tensor_list = torch.split(qkv, self.attention_head_size, 0)
                q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
                q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'h.{i}.self_attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'h.{i}.self_attention.query_key_value.bias'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    qkv.append(state_dict.pop(model_key.format(i, i_k)).split(self.attention_head_size, 0))
                state_dict[ckpt_key] = torch.cat([torch.cat(i) for i in zip(*qkv)])
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{model_key: ckpt_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'word_embeddings.weight',
            'embeddings.layerNorm.weight': 'word_embeddings_layernorm.weight',
            'embeddings.layerNorm.bias': 'word_embeddings_layernorm.bias',
            'lm_head.weight': 'lm_head.weight' if self.with_lm and not self.tie_word_embeddings else 'word_embeddings.weight',
            'LayerNormFinal.weight': 'ln_f.weight',
            'LayerNormFinal.bias': 'ln_f.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'h.{i}.self_attention.dense.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'h.{i}.self_attention.dense.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'h.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'h.{i}.input_layernorm.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'h.{i}.mlp.dense_h_to_4h.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'h.{i}.mlp.dense_h_to_4h.bias',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'h.{i}.mlp.dense_4h_to_h.weight',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'h.{i}.mlp.dense_4h_to_h.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'h.{i}.post_attention_layernorm.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'h.{i}.post_attention_layernorm.bias'
            })
        return mapping
