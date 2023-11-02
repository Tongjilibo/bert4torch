from bert4torch.models.internlm import InternLM
import torch


class Qwen(InternLM):
    '''通义千问: https://github.com/QwenLM/Qwen-7B
    1）FeedForward和Llama一致，三个dense层
    2）除了qkv有bias，其余均没有bias
    3) 和InternLM基本一致，唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = 'qwen'
        for layer in self.decoderLayer:
            layer.multiHeadAttention.o.register_parameter('bias', None)
    '''
    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        qkv = ['q', 'k', 'v']
        for i in range(self.num_hidden_layers):
            old_key = 'transformer.h.%s.attn.c_attn.weight' % i
            if (w := state_dict.get(old_key)) is not None:
                ws = torch.chunk(w, 3, dim=0)
                for k, w in zip(qkv, ws):
                    state_dict[f'decoderLayer.{i}.multiHeadAttention.{k}.weight'] = w
                state_dict.pop(old_key)

            old_key = 'transformer.h.%s.attn.c_attn.bias' % i
            if (b := state_dict.get(old_key)) is not None:
                bs = torch.chunk(b, 3, dim=0)
                for k, b in zip(qkv, bs):
                    state_dict[f'decoderLayer.{i}.multiHeadAttention.{k}.weight'] = b
                state_dict.pop(old_key)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': f'transformer.wte.weight',
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
            f'decoderLayer.{i}.feedForward.outputDense.weight':'transformer.h.%s.mlp.c_proj.weight' % i,
            f'decoderLayer.{i}.ffnLayerNorm.weight': 'transformer.h.%s.ln_2.weight' % i
            })
        return mapping
    '''