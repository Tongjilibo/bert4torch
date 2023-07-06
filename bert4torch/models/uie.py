from bert4torch.models.bert import BERT
from bert4torch.snippets import delete_arguments
from torch import nn
import torch


class UIE(BERT):
    '''官方项目：https://github.com/universal-ie/UIE；
       参考项目：https://github.com/heiheiyoyo/uie_pytorch
    '''
    @delete_arguments('with_nsp', 'with_mlm')
    def __init__(self, *args, **kwargs):
        super(UIE, self).__init__(*args, **kwargs)
        hidden_size = self.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        if kwargs.get('sigmoid', True):
            self.sigmoid = nn.Sigmoid()

        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output+task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def apply_final_layers(self, **model_kwargs):
        hidden_states = super().apply_final_layers(**model_kwargs)  # 仅有hidden_state一项输出
        sequence_output = hidden_states[0] if isinstance(hidden_states, (tuple, list)) else hidden_states

        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits) if hasattr(self, 'sigmoid') else start_logits
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits) if hasattr(self, 'sigmoid') else end_logits

        if isinstance(hidden_states, (tuple, list)):
            return hidden_states + [start_prob, end_prob]
        else:
            return hidden_states, start_prob, end_prob

    def variable_mapping(self):
        mapping = super(UIE, self).variable_mapping()
        mapping.update({'linear_start.weight': 'linear_start.weight',
                        'linear_start.bias': 'linear_start.bias',
                        'linear_end.weight': 'linear_end.weight',
                        'linear_end.bias': 'linear_end.bias'})
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping
