import imp
from bert4torch.models.transformer_xl import Transformer_XL
import torch
import re


class XLNET(Transformer_XL):
    '''构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入；
       接受的inputs输入: [token_ids, segment_ids]
    '''
    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)
    
    def relative_positional_encoding(self, qlen, klen, device):
        # 生成pos_emb, 这里使用sincos的位置编码, transformer_xl里面有-1
        if self.attn_type == 'bi':
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.") 

        # 前向的emb
        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)

        # 双向数据
        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb

        pos_emb = self.dropout(pos_emb)  # 用word_emb的dropout
        return pos_emb

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        if self.with_lm:
            return [hidden_state, self.lm_head(hidden_state)]
        else:
            return hidden_state

    def load_variable(self, state_dict, name, prefix='transformer'):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {f'{prefix}.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        elif re.search('rel_attn\.(q|k|v|r)$', name):
            return variable.reshape(variable.shape[0], -1).T
        # elif re.search('rel_attn\.(o|seg_embed)$', name):
        elif re.search('rel_attn\.(o)$', name):
            return variable.reshape(variable.shape[0], -1)
        else:
            return variable

    def variable_mapping(self, prefix='transformer'):
        mapping = {
            'embeddings.weight': f'{prefix}.word_embedding.weight',
            'lm_head.weight': 'lm_loss.weight',
            'lm_head.bias': 'lm_loss.bias',
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'rel_attn.o',
                            f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r',
                            f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias',
                            # f'encoderLayer.{i}.multiHeadAttention.seg_embed.weight': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'rel_attn.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'rel_attn.layer_norm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'ff.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'ff.layer_norm.bias'
                            })
        return mapping
