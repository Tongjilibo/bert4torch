from bert4torch.models.internlm import InternLM
import torch


class Qwen(InternLM):
    '''通义千问: https://github.com/QwenLM/Qwen-7B
    1) FeedForward和Llama一致, 三个dense层
    2) 除了qkv有bias, 其余均没有bias
    3) 和InternLM基本一致, 唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'qwen'
        for layer in self.decoderLayer:
            layer.multiHeadAttention.o.register_parameter('bias', None)

    def load_trans_ckpt(self, checkpoint):
        '''原始权重中qkv是一个全连接, 需要拆分成q,k,v'''
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            mapping = {
                'transformer.h.%s.attn.c_attn.weight' % i: 'decoderLayer.{}.multiHeadAttention.{}.weight',
                'transformer.h.%s.attn.c_attn.bias' % i: 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for old_key, new_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(old_key)) is None:
                    continue
                qkv = torch.chunk(qkv, 3, dim=0)
                for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                    state_dict[new_key.format(i, i_k)] = i_v
                state_dict.pop(old_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': 'transformer.h.%s.attn.c_attn.weight' % i,
                'decoderLayer.{}.multiHeadAttention.{}.bias': 'transformer.h.%s.attn.c_attn.bias' % i
            }
            for old_key, new_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    qkv.append(state_dict.pop(old_key.format(i, i_k)))
                if qkv:
                    state_dict[new_key] = torch.cat(qkv)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典, 格式为{new_key: old_key}"""
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
