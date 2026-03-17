from ..base import Decoder, register_model
from bert4torch.layers import NormHead


@register_model(name="llama")
class LLaMA(Decoder):
    '''LLaMA
    链接: https://github.com/facebookresearch/llama
    1. 去掉bias
    2. rmsnorm
    3. feedForward不同, 三层全连接
    4. rotary相对位置编码
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # 修改lm_head，目前在Baichuan2中使用
        if kwargs.get('norm_head') is True:
            self.lm_head = NormHead(self.hidden_size, self.vocab_size)
    
    def variable_mapping(self):
        '''映射到权重格式
        llama一般有两种格式, 一种是huggingface格式, 一种是pth格式, 这里的映射是以hf格式为准
        '''
        mapping = {
            'embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'lm_head.weight': 'lm_head.weight' if self.with_lm and not self.tie_word_embeddings else 'model.embed_tokens.weight',
            'LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.q.weight': f'model.layers.{i}.self_attn.q_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'model.layers.{i}.self_attn.k_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'model.layers.{i}.self_attn.v_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.self_attn.o_proj.weight',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping

