from bert4torch.models.transformer_xl import Transformer_XL
import torch
from torch4keras.snippets import safe_torch_load


class XLNET(Transformer_XL):
    '''构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入；
       接受的inputs输入: [token_ids, segment_ids]
    '''
    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)
        self.model_type = 'xlnet'
    
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
        outputs = super().apply_final_layers(**model_kwargs)
        last_hidden_state = outputs['last_hidden_state'] if self.return_dict else outputs

        if self.with_lm:
            logits = self.lm_head(last_hidden_state)
            return self.gen_outputs(locals(), last_hidden_state, logits) if self.return_dict else [last_hidden_state, logits]
        elif not self.return_dict:
            return last_hidden_state
        else:
            return self.gen_outputs(locals(), last_hidden_state)

    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {f'transformer.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def load_trans_ckpt(self, checkpoint):
        state_dict = safe_torch_load(checkpoint, map_location='cpu')
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping = {
                    prefix_i + 'rel_attn.q': f'encoderLayer.{i}.multiHeadAttention.q.weight',
                    prefix_i + 'rel_attn.k': f'encoderLayer.{i}.multiHeadAttention.k.weight',
                    prefix_i + 'rel_attn.v': f'encoderLayer.{i}.multiHeadAttention.v.weight',
                    prefix_i + 'rel_attn.r': f'encoderLayer.{i}.multiHeadAttention.r.weight',
                    }
            for ckpt_key, model_key in mapping.items():
                if state_dict.get(ckpt_key) is not None:
                    variable = state_dict.pop(ckpt_key)
                    state_dict[model_key] = variable.reshape(variable.shape[0], -1).T

            ckpt_key = prefix_i + 'rel_attn.o'
            if state_dict.get(ckpt_key) is not None:
                variable = state_dict.pop(ckpt_key)
                state_dict[f'encoderLayer.{i}.multiHeadAttention.o.weight'] = variable.reshape(variable.shape[0], -1)
        return state_dict

    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping = {
                    f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q',
                    f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k',
                    f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v',
                    f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r',
                    }
            for model_key, ckpt_key in mapping.items():
                if state_dict.get(model_key) is not None:
                    variable = state_dict.pop(model_key)
                    state_dict[ckpt_key] = variable.T.reshape(-1, self.num_attention_heads, self.attention_head_size)

            model_key = f'encoderLayer.{i}.multiHeadAttention.o.weight'
            if state_dict.get(model_key) is not None:
                variable = state_dict.pop(model_key)
                state_dict[prefix_i + 'rel_attn.o'] = variable.reshape(-1, self.num_attention_heads, self.attention_head_size)
        return state_dict

    def variable_mapping(self):
        mapping = {
            'embeddings.weight': f'transformer.word_embedding.weight',
            'lm_head.weight': 'lm_loss.weight',
            'lm_head.bias': 'lm_loss.bias',
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping.update({
                            f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias',
                            # f'encoderLayer.{i}.multiHeadAttention.seg_embed.weight': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'rel_attn.layer_norm.weight',
                            f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'rel_attn.layer_norm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias',
                            f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'ff.layer_norm.weight',
                            f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'ff.layer_norm.bias'
                            })
        return mapping
