from bert4torch.models.bert import BERT
from bert4torch.snippets import delete_arguments
from bert4torch.layers import ConvLayer


class DebertaV2(BERT):
    '''DeBERTaV2: https://arxiv.org/abs/2006.03654, https://github.com/microsoft/DeBERTa:
       这里使用的是IEDEA的中文模型: https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese:
       和transformers包中的区别: 
       1) 原始使用的StableDropout替换成了nn.Dropout:
       2) 计算attention_score时候用的XSoftmax替换成了F.softmax:
       3) 未实现(认为对结果无影响): Embedding阶段用attention_mask对embedding的padding部分置0:
       4) 未实现(认为对结果无影响): 计算attention_score前的attention_mask从[btz, 1, 1, k_len]转为了[btz, 1, q_len, k_len]。
    '''
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'deberta_v2'})  # 控制在Embedding阶段不生成position_embedding
        super(DebertaV2, self).__init__(*args, **kwargs)
        self.model_type = 'deberta'
        # Encoder中transformer_block前的其他网络结构
        self.relative_attention = kwargs.get("relative_attention", True)
        self.conv = ConvLayer(**kwargs) if kwargs.get("conv_kernel_size", 0) > 0 else None
        self.tie_weights()
    
    def tie_weights(self):
        super().tie_weights()
        # 把第二层后的相对位置编码的权重绑定到第一层上, 变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.weight = self.encoderLayer[0].multiHeadAttention.layernorm.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.bias = self.encoderLayer[0].multiHeadAttention.layernorm.bias

    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for i in range(1, self.num_hidden_layers):
            state_dict.pop(f'encoderLayer.{i}.multiHeadAttention.relative_positions_encoding.weight')
            state_dict.pop(f'encoderLayer.{i}.multiHeadAttention.relative_positions.relative_position')
            state_dict.pop(f'encoderLayer.{i}.multiHeadAttention.layernorm.weight')
            state_dict.pop(f'encoderLayer.{i}.multiHeadAttention.layernorm.bias')
        return state_dict
    
    def apply_main_layers(self, **model_kwargs):
        """DebertaV2: 主要区别是第0层后, 会通过卷积层"""
        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.encoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            # 第0层要经过卷积
            if l_i == 0 and self.conv is not None:
                model_kwargs['hidden_states'] = self.conv(encoded_layers[0], model_kwargs['hidden_states'], model_kwargs['attention_mask'].squeeze(1).squeeze(1))
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                encoded_layers.append(model_kwargs['hidden_states'])
        if not self.output_all_encoded_layers:
            encoded_layers.append(model_kwargs['hidden_states'])
        model_kwargs['encoded_layers'] =  encoded_layers
        return model_kwargs

    def variable_mapping(self):
        mapping = super(DebertaV2, self).variable_mapping(prefix='deberta')
        mapping.update({'mlmDecoder.weight': 'deberta.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias',
                        'encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'deberta.encoder.rel_embeddings.weight',
                        'encoderLayer.0.multiHeadAttention.layernorm.weight': 'deberta.encoder.LayerNorm.weight',
                        'encoderLayer.0.multiHeadAttention.layernorm.bias': 'deberta.encoder.LayerNorm.bias',
                        'conv.conv.weight': 'deberta.encoder.conv.conv.weight',
                        'conv.conv.bias': 'deberta.encoder.conv.conv.bias',
                        'conv.LayerNorm.weight': 'deberta.encoder.conv.LayerNorm.weight',
                        'conv.LayerNorm.bias': 'deberta.encoder.conv.LayerNorm.bias'})
        for del_key in ['nsp.weight', 'nsp.bias', 'embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight']:
            del mapping[del_key]
        
        # 把ckpt_key中的部分关键字替换一下
        rep_str = {'query': 'query_proj', 'key': 'key_proj', 'value': 'value_proj'}
        mapping_tmp = {}
        for model_key, ckpt_key in mapping.items():
            for i, k in rep_str.items():
                if i in ckpt_key:
                    mapping_tmp[model_key] = ckpt_key.replace(i, k)
        mapping.update(mapping_tmp)
        return mapping
