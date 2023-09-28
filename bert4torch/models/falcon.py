from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import MultiHeadAttentionLayer
import math
import torch

class Falcon(Decoder):
    '''Falcon: https://huggingface.co/tiiuae
    主要区别就是alibi编码，但是其attention_scale是在+attention_mask后执行的，和bloom、baichuan-13b-chat其他不一样
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'pre_layernorm': True, 'norm_mode': 'torch_buildin', 
                      'is_decoder': True, 'final_layernorm': True, 'attention_scale': False})
        MultiHeadAttentionLayer.apply_alibi_pos_emb = apply_alibi_pos_emb
        super().__init__(*args, **kwargs)
        self.prefix = 'falcon'
        del self.embeddings.layerNorm
        
def apply_alibi_pos_emb(self, attention_scores, key_layer):
    ''' 执行alibi相对位置编码，单独拎出来主要是falcon是在+之后再执行attention_scale的 '''
    if (self.p_bias == 'alibi') and hasattr(self, 'relative_positions_encoding'):
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
            
        key_position_scores_r_t = self.relative_positions_encoding(key_layer)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    return attention_scores
