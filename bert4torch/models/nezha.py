from bert4torch.models.bert import BERT


class NEZHA(BERT):
    """华为推出的NAZHA模型；
    链接：https://arxiv.org/abs/1909.00204
    """
    def __init__(self, *args, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding, max_relative_position默认取64
        kwargs.update({'p_bias': 'typical_relative', 'max_relative_position': kwargs.get('max_relative_position', 64)})
        super(NEZHA, self).__init__(*args, **kwargs)
