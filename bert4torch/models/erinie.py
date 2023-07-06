from bert4torch.models.bert import BERT


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""
    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)

    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if ('LayerNorm.weight' in v) or ('LayerNorm.bias' in v):
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='ernie'):
        return super().load_variable(state_dict, name, prefix=prefix)
