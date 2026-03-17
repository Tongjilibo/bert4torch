from ..base import Encoder, Decoder, Transformer, register_model
from bert4torch.snippets import insert_arguments, delete_arguments
from bert4torch.layers import RMSNorm
from torch import nn


@register_model(name='mt5.1.1_encoder')
@register_model(name='t5.1.0_encoder')
@register_model(name='t5.1.1_encoder')
@register_model(name='t5_encoder')
class T5_Encoder(Encoder):
    _no_split_modules = ["T5Layer"]
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, version=self.version, **kwargs)
        del self.embeddings.layerNorm

        self.final_layer_norm = RMSNorm(self.hidden_size, layer_norm_eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.tie_weights()

    def tie_weights(self):
        super().tie_weights()
        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight

    def apply_final_layers(self, **model_kwargs):
        outputs = super().apply_final_layers(**model_kwargs)
        last_hidden_state = outputs['last_hidden_state'] if self.return_dict else outputs
        return self.dropout(self.final_layer_norm(last_hidden_state))

    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {f'encoder.embed_tokens.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        # 查看check_point发现'shared.weight'
        mapping = {f'encoder.embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight',
                   f'encoder.encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'encoder.final_layer_norm.weight': 'encoder.final_layer_norm.weight'}
        for i in range(self.num_hidden_layers):
            mapping.update({
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'encoder.encoderLayer.{i}.attnLayerNorm.weight': f'encoder.block.{i}.layer.0.layer_norm.weight',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight',
                f'encoder.encoderLayer.{i}.ffnLayerNorm.weight': f'encoder.block.{i}.layer.1.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight',
                                f'encoder.encoderLayer.{i}.feedForward.intermediateDense1.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'})
        return mapping
    

@register_model(name='mt5.1.1_decoder')
@register_model(name='t5.1.0_decoder')
@register_model(name='t5.1.1_decoder')
@register_model(name='t5_decoder')
class T5_Decoder(Decoder):
    _no_split_modules = ["T5Layer"]
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, version=self.version, **kwargs)
        del self.embeddings.layerNorm        
        self.final_layer_norm = RMSNorm(self.hidden_size, layer_norm_eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.tie_weights()

    def tie_weights(self):
        super().tie_weights()
        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.decoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.decoderLayer[0].multiHeadAttention.relative_positions_encoding.weight

    def apply_final_layers(self, **model_kwargs):
        # 这里的encoded_layers没有改成decoded_layers是想使用super()
        last_hidden_state = model_kwargs['decoded_layers'][-1]
        model_kwargs['decoded_layers'][-1] = self.dropout(self.final_layer_norm(last_hidden_state))  # 在转logit前把最后一层的hidden_states加layernorm
        return super().apply_final_layers(**model_kwargs)

    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {f'decoder.embed_tokens.weight', 'lm_head.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        # 查看check_point发现'shared.weight'
        mapping = {f'decoder.embeddings.word_embeddings.weight': 'decoder.embed_tokens.weight',
                   f'decoder.decoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'decoder.final_layer_norm.weight': 'decoder.final_layer_norm.weight',
                   f'decoder.lm_head.weight': 'lm_head.weight'}

        for i in range(self.num_hidden_layers):
            mapping.update({
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'decoder.decoderLayer.{i}.attnLayerNorm.weight': f'decoder.block.{i}.layer.0.layer_norm.weight',

                f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.block.{i}.layer.1.EncDecAttention.q.weight',
                f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.block.{i}.layer.1.EncDecAttention.k.weight',
                f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.block.{i}.layer.1.EncDecAttention.v.weight',
                f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.block.{i}.layer.1.EncDecAttention.o.weight',
                f'decoder.decoderLayer.{i}.crossLayerNorm.weight': f'decoder.block.{i}.layer.1.layer_norm.weight',

                f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight',
                f'decoder.decoderLayer.{i}.ffnLayerNorm.weight': f'decoder.block.{i}.layer.2.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight',
                                f'decoder.decoderLayer.{i}.feedForward.intermediateDense1.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'})
        return mapping


@register_model(name='mt5.1.1')
@register_model(name='t5')
@register_model(name='t5.1.0')
@register_model(name='t5.1.1')
class T5(Transformer):
    """Google的T5模型: Encoder-Decoder
    decoder: tie_word_embeddings=False
    encoder-decoder: tie_word_embeddings_encoder_decoder=True
    """
    def __init__(self, *args,  tie_word_embeddings_encoder_decoder:bool=True, **kwargs):
        kwargs['tie_word_embeddings_encoder_decoder'] = tie_word_embeddings_encoder_decoder
        super(T5, self).__init__(*args, **kwargs)

        # encoder
        self.encoder = T5_Encoder(*args, **kwargs)

        # decoder
        kwargs['add_cross_attention'] = True
        kwargs['logit_scale'] = kwargs.get('logit_scale', True)
        self.decoder = T5_Decoder(*args, **kwargs)
    
    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        mapping = self.encoder.variable_mapping()
        mapping.update(self.decoder.variable_mapping())
        if self.tie_word_embeddings_encoder_decoder:
            mapping.update({'encoder.embeddings.word_embeddings.weight': 'shared.weight',
                            'decoder.embeddings.word_embeddings.weight': 'shared.weight'})
        return mapping
