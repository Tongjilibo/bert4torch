from bert4torch.models.base import BertBase, register_model
from bert4torch.layers import ErnieEmbeddings


@register_model(name="ernie")
class Ernie(BertBase):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""
    def __init__(self, *args, **kwargs):
        super(Ernie, self).__init__(*args, **kwargs)
        self.use_task_id = kwargs.get('use_task_id')
        self.embeddings = ErnieEmbeddings(**self.get_kw(*self._embedding_args, **kwargs))

    def variable_mapping(self):
        mapping = super(Ernie, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if ('LayerNorm.weight' in v) or ('LayerNorm.bias' in v):
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        
        if self.use_task_id:
            mapping['embeddings.task_type_embeddings.weight'] = 'ernie.embeddings.task_type_embeddings.weight'
        return mapping