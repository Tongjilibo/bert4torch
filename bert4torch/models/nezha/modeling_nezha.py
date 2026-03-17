from ..base import BertBase, register_model
from bert4torch.snippets import modify_variable_mapping


@register_model(name="nezha")
class NEZHA(BertBase):
    """华为推出的NAZHA模型；
    链接：https://arxiv.org/abs/1909.00204
    """
    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        mapping = {}
        if ('cls.predictions.bias' in state_dict) and ('cls.predictions.decoder.bias' not in state_dict):
            mapping['mlmDecoder.bias'] = 'cls.predictions.bias'
        self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict