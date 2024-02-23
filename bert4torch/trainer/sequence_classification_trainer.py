import torch
from torch import nn
from torch4keras.model import BaseModel
from bert4torch.snippets import get_pool_emb


class SequenceClassificationTrainer(BaseModel):
    def __init__(self, model, num_labels=2, classifier_dropout=None, pool_strategy='cls', **kwargs):
        super().__init__()
        self.model = model
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