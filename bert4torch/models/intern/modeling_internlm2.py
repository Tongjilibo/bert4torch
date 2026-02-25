from bert4torch.models.base import Decoder, register_model
import torch


@register_model(name="internlm2")
class InternLM2(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

    def load_trans_ckpt(self, checkpoint):
        '''原始权重中qkv是一个全连接, 需要拆分成q,k,v'''
        state_dict = super().load_trans_ckpt(checkpoint)
        # qkv有特殊的排布方式
        for i in range(self.num_hidden_layers):
            ckpt_key = 'model.layers.%s.attention.wqkv.weight' % i
            model_key = 'decoderLayer.{}.multiHeadAttention.{}.weight'
            # 如果当前ckpt不存在该key，则跳过
            if (qkv := state_dict.get(ckpt_key)) is None:
                continue
            num_key_value_groups = self.num_attention_heads // self.decoderLayer[0].multiHeadAttention.num_key_value_heads
            num_key_value_heads = qkv.shape[0] // (2 + num_key_value_groups) // self.attention_head_size
            qkv = qkv.reshape(num_key_value_heads, -1, self.attention_head_size, qkv.shape[-1])
            q = qkv[:, :num_key_value_groups, :, :].reshape(-1, qkv.shape[-1])
            k = qkv[:, -2, :, :].reshape(-1, qkv.shape[-1])
            v = qkv[:, -1, :, :].reshape(-1, qkv.shape[-1])
            for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                state_dict[model_key.format(i, i_k)] = i_v
            state_dict.pop(ckpt_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            ckpt_key = 'model.layers.%s.attention.wqkv.weight' % i
            model_key = 'decoderLayer.{}.multiHeadAttention.{}.weight'
            k = state_dict.pop(model_key.format(i, 'k'))
            num_key_value_heads = k.shape[0] // self.attention_head_size
            k = k.reshape(num_key_value_heads, -1, self.attention_head_size, k.shape[-1])
            v = state_dict.pop(model_key.format(i, 'v'))
            v = v.reshape(num_key_value_heads, -1, self.attention_head_size, v.shape[-1])
            q = state_dict.pop(model_key.format(i, 'q'))
            q = q.reshape(num_key_value_heads, -1, self.attention_head_size, q.shape[-1])
            state_dict[ckpt_key] = torch.cat([q, k, v], dim=1).reshape(-1, self.hidden_size)
        return state_dict
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'embeddings.word_embeddings.weight': 'model.tok_embeddings.weight',
            'lm_head.weight': 'output.weight' if self.with_lm and not self.tie_word_embeddings else 'model.tok_embeddings.weight',
            'LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.attention.wo.weight',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.attention_norm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.feed_forward.w1.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.feed_forward.w3.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.feed_forward.w2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.ffn_norm.weight'
            })
        return mapping