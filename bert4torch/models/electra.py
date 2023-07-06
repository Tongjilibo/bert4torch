from bert4torch.models.bert import BERT
from bert4torch.snippets import insert_arguments, delete_arguments
from torch import nn
from bert4torch.activations import get_activation


class ELECTRA(BERT):
    """Google推出的ELECTRA模型；
    链接：https://arxiv.org/abs/2003.10555
    """
    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = get_activation('sigmoid') if self.with_discriminator is True else get_activation(self.with_discriminator)

    def apply_final_layers(self, **model_kwargs):
        hidden_states = super().apply_final_layers(**model_kwargs)  # 仅有hidden_state一项输出
        if self.with_discriminator:
            logit = self.dense_act(self.dense(hidden_states))
            return [hidden_states, self.dense_prediction_act(self.dense_prediction(logit))]
        else:
            return hidden_states

    def load_variable(self, state_dict, name):
        # 加载单个变量的函数
        return super().load_variable(state_dict, name, prefix='electra')

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix='electra')
        mapping.update({'dense.weight': 'discriminator_predictions.dense.weight', 
                        'dense.bias': 'discriminator_predictions.dense.bias',
                        'dense_prediction.weight': 'discriminator_predictions.dense_prediction.weight',
                        'dense_prediction.bias': 'discriminator_predictions.dense_prediction.bias'}
                        )
        for del_key in ['pooler.weight', 'pooler.bias', 'nsp.weight', 'nsp.bias', 'mlmDense.weight', 'mlmDense.bias', 
                        'mlmLayerNorm.weight', 'mlmLayerNorm.bias', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias']:
            del mapping[del_key]

        return mapping