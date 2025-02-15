from bert4torch.models.bert import BERT
from torch import nn
import copy


class ALBERT(BERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])  # 取上述的第一行
        self.model_type = 'albert'

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和BERT区别是始终使用self.encoderLayer[0]）；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """

        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i in range(self.num_hidden_layers):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            layer_module = self.encoderLayer[0]
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] =  encoded_layers
        return model_kwargs

    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'albert.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': 'albert.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'albert.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': 'albert.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': 'albert.embeddings.LayerNorm.bias',
            'embeddings.embedding_hidden_mapping_in.weight': 'albert.encoder.embedding_hidden_mapping_in.weight',
            'embeddings.embedding_hidden_mapping_in.bias': 'albert.encoder.embedding_hidden_mapping_in.bias',
            'pooler.weight': 'albert.pooler.weight',
            'pooler.bias': 'albert.pooler.bias',
            'nsp.weight': 'sop_classifier.classifier.weight',  # 用名字nsp来替换sop
            'nsp.bias': 'sop_classifier.classifier.bias',
            'mlmDense.weight': 'predictions.dense.weight',
            'mlmDense.bias': 'predictions.dense.bias',
            'mlmLayerNorm.weight': 'predictions.LayerNorm.weight',
            'mlmLayerNorm.bias': 'predictions.LayerNorm.bias',
            'mlmBias': 'predictions.bias',
            'mlmDecoder.weight': 'predictions.decoder.weight',
            'mlmDecoder.bias': 'predictions.decoder.bias'
        }
        i = 0
        prefix_i = f'albert.encoder.albert_layer_groups.{i}.albert_layers.{i}.'
        mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.query.weight',
                        f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.query.bias',
                        f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.key.weight',
                        f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.key.bias',
                        f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.value.weight',
                        f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.value.bias',
                        f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight',
                        f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias',
                        f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'attention.LayerNorm.weight',
                        f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'attention.LayerNorm.bias',
                        f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ffn.weight',
                        f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ffn.bias',
                        f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ffn_output.weight',
                        f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ffn_output.bias',
                        f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'full_layer_layer_norm.weight',
                        f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'full_layer_layer_norm.bias'
                        })

        return mapping

    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {
            'albert.embeddings.word_embeddings.weight',
            'predictions.bias',
            'predictions.decoder.weight',
            'predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif ckpt_key == 'sop_classifier.classifier.weight':
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)])
        self.model_type = 'albert_unshared'
        
    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和ALBERT区别是所有层权重独立）；这里就是调用BERT类的方法
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        return BERT.apply_main_layers(self, **model_kwargs)

    def variable_mapping(self):
        mapping = super().variable_mapping()
        prefix_0 = f'albert.encoder.albert_layer_groups.0.albert_layers.0.'
        for i in range(1, self.num_hidden_layers):
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_0 + 'attention.query.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_0 + 'attention.query.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_0 + 'attention.key.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_0 + 'attention.key.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_0 + 'attention.value.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_0 + 'attention.value.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_0 + 'attention.dense.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_0 + 'attention.dense.bias',
                            f'encoderLayer.{i}.attnLayerNorm.weight': prefix_0 + 'attention.LayerNorm.weight',
                            f'encoderLayer.{i}.attnLayerNorm.bias': prefix_0 + 'attention.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_0 + 'ffn.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_0 + 'ffn.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_0 + 'ffn_output.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_0 + 'ffn_output.bias',
                            f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_0 + 'full_layer_layer_norm.weight',
                            f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_0 + 'full_layer_layer_norm.bias'
                            })
        return mapping