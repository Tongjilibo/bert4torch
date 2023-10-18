from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import MultiHeadAttentionLayer, BertLayer, BlockIdentity
import math
import torch
from torch import nn
import copy


class Falcon(Decoder):
    '''Falcon: https://huggingface.co/tiiuae
    falcon-rw-1b：主要区别就是alibi编码，但是其attention_scale是在+attention_mask后执行的，和bloom、baichuan-13b-chat其他不一样
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'weight': True, 'pre_layernorm': True, 'norm_mode': 'torch_buildin', 
                      'is_decoder': True, 'final_layernorm': True, 'attention_scale': False})
        if kwargs.get('p_bias') == 'alibi':
            MultiHeadAttentionLayer.apply_alibi_pos_emb = apply_alibi_pos_emb
        super().__init__(*args, **kwargs)
        self.prefix = 'falcon'
        del self.embeddings.layerNorm

        if kwargs.get('parallel_attn') is True:
            layer = self.ParallelAttnLayer(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                            'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'max_position', **kwargs))
            self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
            self.LayerNormFinal.bias = nn.Parameter(torch.zeros(kwargs['hidden_size']))

    class ParallelAttnLayer(BertLayer):
        ''''''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layerNorm1.bias = nn.Parameter(torch.zeros(kwargs['hidden_size']))
            del self.layerNorm2

        def forward(self, hidden_states=None, attention_mask=None, position_ids=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # ============== self attention ==============
            x = self.layerNorm1(hidden_states, conditional_emb)
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)  # self.decoder为true时候，这里的attention_mask是三角的
            
            # ============== feedforward ==============
            feedforward_output = self.feedForward(x)
            feedforward_output += self_attn_output[0]
            hidden_states = self.dropout_add(feedforward_output, hidden_states)
            hidden_states = self.layerNorm2(hidden_states, conditional_emb) if not self.pre_layernorm else hidden_states
            
            if self.is_decoder and model_kwargs.get('use_states', False):
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs


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


