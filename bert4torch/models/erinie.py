from bert4torch.models.bert import BERT
import torch


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""
    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)
        self.prefix = 'ernie'

    def load_trans_ckpt(self, checkpoint):
        return torch.load(checkpoint, map_location='cpu')
    
    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping()
        mapping.update({'mlmDecoder.weight': f'{self.prefix}.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if ('LayerNorm.weight' in v) or ('LayerNorm.bias' in v):
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping
