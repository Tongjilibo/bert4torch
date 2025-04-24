'''
sequence classification trainer
'''
from torch import nn
from torch4keras.model import BaseModel
from torch4keras.trainer import AutoTrainer, Trainer
from bert4torch.snippets import get_pool_emb
from typing import Literal, Union


class SequenceClassificationModel(BaseModel):
    def __init__(self, module:nn.Module, num_labels:int=2, classifier_dropout:float=None, 
                 pool_strategy:Literal['pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom']='cls', **kwargs):
        ''' 文本分类Model
        :param module: 预训练模型的Module
        :param num_labels: int, 文本分类的类型数
        :param classifier_dropout: float, dropout比例
        :param pool_strategy: str, 选取句向量的策略, 默认为cls
        :param kwargs: dict, 其他build_transformer_model使用到的参数
        '''
        super().__init__()
        module.return_dict = True  # 返回的字典
        self.model = module
        self.config = self.model.config
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.num_labels = num_labels
        self.pool_strategy = pool_strategy
        if classifier_dropout:
            self.dropout = nn.Dropout(classifier_dropout)
        elif hasattr(self.config, "hidden_dropout_prob"):
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        else:
            self.dropout = lambda x: x
        self.classifier = nn.Linear(self.config.hidden_size, num_labels, dtype=module.dtype)

    def forward(self, *args, **kwarg):
        output = self.model(*args, **kwarg)

        if len(args) > 0:
            attention_mask = (args[0] != self.pad_token_id).long()
        elif (input_ids := kwarg.get('input_ids') or kwarg.get('token_ids')) is not None:
            attention_mask = (input_ids != self.pad_token_id).long()
        else:
            raise TypeError('Args `batch_input` only support list(tensor)/tensor format')

        last_hidden_state = output.get('last_hidden_state')
        pooler = output.get('pooled_output')

        pooled_output = get_pool_emb(last_hidden_state, pooler, attention_mask, self.pool_strategy)
        output = self.classifier(self.dropout(pooled_output))  # [btz, num_labels]
        return output


class SequenceClassificationTrainer(AutoTrainer):
    '''文本分类的Trainer
    :param module: 预训练模型的Module
    :param num_labels: int, 文本分类的类型数
    :param classifier_dropout: float, dropout比例
    :param pool_strategy: str, 选取句向量的策略, 默认为cls
    :param kwargs: dict, 其他build_transformer_model使用到的参数

    Examples
    ```python
    >>> from bert4torch.trainer import SequenceClassificationTrainer
    >>> from bert4torch.models import build_transformer_model
    >>> config_path = ''  # bert4torch_config.json路径
    >>> checkpoint_path = ''  # 模型文件夹路径
    >>> bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
    >>> model = SequenceClassificationTrainer(bert).to('cuda')
    ```
    '''
    def __init__(self, module:BaseModel, *args, num_labels:int=2, classifier_dropout:float=None, 
                pool_strategy:Literal['pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom']='cls', **kwargs):
        pass

    def __new__(cls, module:BaseModel, *args, num_labels:int=2, classifier_dropout:float=None, 
                pool_strategy:Literal['pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom']='cls', **kwargs) -> Trainer:
        model = SequenceClassificationModel(module, num_labels, classifier_dropout, pool_strategy, **kwargs)
        model.to(module.device)
        return super().__new__(cls, model, *args, **kwargs)