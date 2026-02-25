from bert4torch.models.base import BertBase, register_model
from bert4torch.snippets import modify_variable_mapping


@register_model(name="bert")
@register_model(name="roberta")
class BERT(BertBase):
    def load_trans_ckpt(self, checkpoint):
        """加载ckpt, 方便后续继承并做一些预处理
        这么写的原因是下游很多模型从BERT继承，这样下游可以默认使用PreTrainedModel的load_trans_ckpt
        """
        state_dict = super().load_trans_ckpt(checkpoint)        
        mapping_reverse = {v:k for k, v in self.variable_mapping().items()}
        mapping = {}
        for key in state_dict.keys():
            # bert-base-chinese中ln的weight和bias是gamma和beta
            if ".gamma" in key:
                value = key.replace(".gamma", ".weight")
                mapping[mapping_reverse[value]] = key
            if ".beta" in key:
                value = key.replace(".beta", ".bias")
                mapping[mapping_reverse[value]] = key
        if ('cls.predictions.bias' in state_dict) and ('cls.predictions.decoder.bias' not in state_dict):
            mapping['mlmDecoder.bias'] = 'cls.predictions.bias'
        self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict