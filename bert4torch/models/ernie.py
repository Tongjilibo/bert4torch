from bert4torch.models.bert import BERT
from bert4torch.layers import BertEmbeddings
from torch import nn
import torch


class Ernie(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""
    def __init__(self, *args, **kwargs):
        super(Ernie, self).__init__(*args, **kwargs)
        self.use_task_id = kwargs.get('use_task_id')
        self.embeddings = self.ErnieEmbeddings(**self.get_kw(*self._embedding_args, **kwargs))
        self.model_type = 'ernie'

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

    class ErnieEmbeddings(BertEmbeddings):
        def __init__(self, vocab_size, embedding_size, *args, **kwargs):
            super().__init__(vocab_size, embedding_size, *args, **kwargs)
            self.use_task_id = kwargs.get('use_task_id')

            if self.use_task_id:
                self.task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), embedding_size)
        
        def apply_embeddings(self, token_ids, segment_ids, position_ids, additional_embs, **kwargs):
            embeddings = super().apply_embeddings(token_ids, segment_ids, position_ids, additional_embs, **kwargs)

            task_type_ids = kwargs.get('task_type_ids')
            if self.use_task_id:
                if task_type_ids is None:
                    task_type_ids = torch.zeros(token_ids.shape, dtype=torch.long, device=embeddings.device)
                task_type_embeddings = self.task_type_embeddings(task_type_ids)
                embeddings += task_type_embeddings
            return embeddings