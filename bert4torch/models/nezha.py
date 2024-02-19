from bert4torch.models.bert import BERT
from bert4torch.snippets import modify_variable_mapping


class NEZHA(BERT):
    """华为推出的NAZHA模型；
    链接：https://arxiv.org/abs/1909.00204
    """
    def __init__(self, *args, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding, max_relative_position默认取64
        kwargs.update({'p_bias': 'typical_relative', 'max_relative_position': kwargs.get('max_relative_position', 64)})
        super(NEZHA, self).__init__(*args, **kwargs)
        self.model_type = 'nezha'

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        mapping = {}
        if ('cls.predictions.bias' in state_dict) and ('cls.predictions.decoder.bias' not in state_dict):
            mapping['mlmDecoder.bias'] = 'cls.predictions.bias'
        self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict