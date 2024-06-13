'''
sequence classification trainer
'''
from torch import nn
from torch4keras.model import BaseModel
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_pool_emb
from typing import Literal, Union


class SequenceClassificationTrainer(BaseModel):
    def __init__(self, pretrained_model_or_name_path:Union[BaseModel, str], num_labels:int=2, classifier_dropout:float=None, 
                 pool_strategy:Literal['pooler', 'cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom']='cls', **kwargs):
        ''' 文本分类模型
        :param pretrained_model_or_name_path: BaseModel/str, 预训练模型的Module, model_name, model_path
        :param num_labels: int, 文本分类的类型数
        :param classifier_dropout: float, dropout比例
        :param pool_strategy: str, 选取句向量的策略, 默认为cls
        :param kwargs: dict, 其他build_transformer_model使用到的参数
        '''
        super().__init__()
        if isinstance(pretrained_model_or_name_path, str):
            # 从model_path或者model_name加载
            self.model = build_transformer_model(pretrained_model_or_name_path, checkpoint_path=pretrained_model_or_name_path, return_dict=True, **kwargs)
        else:
            # 传入的就是BaseModel
            self.model = pretrained_model_or_name_path
        self.config = self.model.config
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.num_labels = num_labels
        self.pool_strategy = pool_strategy
        self.dropout = nn.Dropout(classifier_dropout or self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

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