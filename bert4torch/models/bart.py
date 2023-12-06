from bert4torch.models.transformer import Transformer
import torch


class BART(Transformer):
    '''encoder-decoder结构'''
    def __init__(self, *args, **kwargs):
        kwargs['logit_scale'] = kwargs.get('logit_scale', False)
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(BART, self).__init__(*args, **kwargs)

    def load_variable(self, state_dict, name):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {
            'shared.weight',
            'encoder.embed_tokens.weight',
            'decoder.embed_tokens.weight',
        }:
            return self.load_embeddings(variable)
        elif name in {'encoder.embed_positions.weight', 'decoder.embed_positions.weight'}:
            return self.load_pos_embeddings(variable)
        else:
            return variable
    
    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        state_dict_new = {}
        for k, v in state_dict.items():
            # V2.0: 这两个权重丢弃，因为一个为0，一个和decoder的embedding一样
            if k in {'final_logits_bias', 'lm_head.weight'}:
                continue
            k = k.replace('model.', '')

            # 主要变更就是默认有514个位置，舍弃前两个位置
            if 'embed_positions.weight' in k:
                v = v[2:]
                state_dict_new[k] = v
            else:
                state_dict_new[k] = v
        del state_dict
        return state_dict_new
     
    def variable_mapping(self):
        # 查看check_point发现'shared.weight'
        mapping = {
            'encoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'encoder.embed_tokens.weight',
            'encoder.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
            'encoder.embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight',
            'encoder.embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias',
            'decoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'decoder.embed_tokens.weight',
            'decoder.embeddings.position_embeddings.weight': 'decoder.embed_positions.weight',
            'decoder.embeddings.layerNorm.weight': 'decoder.layernorm_embedding.weight',
            'decoder.embeddings.layerNorm.bias': 'decoder.layernorm_embedding.bias',
        }
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias',
                f'encoder.encoderLayer.{i}.attnLayerNorm.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight',
                f'encoder.encoderLayer.{i}.attnLayerNorm.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias',
                f'encoder.encoderLayer.{i}.ffnLayerNorm.weight': f'encoder.layers.{i}.final_layer_norm.weight',
                f'encoder.encoderLayer.{i}.ffnLayerNorm.bias': f'encoder.layers.{i}.final_layer_norm.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.attnLayerNorm.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.attnLayerNorm.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.crossLayerNorm.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.crossLayerNorm.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias',
                f'decoder.decoderLayer.{i}.ffnLayerNorm.weight': f'decoder.layers.{i}.final_layer_norm.weight',
                f'decoder.decoderLayer.{i}.ffnLayerNorm.bias': f'decoder.layers.{i}.final_layer_norm.bias'
                })

        return mapping