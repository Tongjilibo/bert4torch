import torch
from torch4keras.trainer import Trainer
from bert4torch.models import BaseModel


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.shape_4 = config.hidden_size // config.num_attention_heads
        if config.get('multi_query_group_num') is not None:
            self.shape_3 = config.multi_query_group_num
            embed_size = self.shape_3 * self.shape_4
        else:
            self.shape_3 = config.num_attention_heads
            embed_size = config.hidden_size
        
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_hidden_layers * embed_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * embed_size * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class PtuningV2Trainer(BaseModel):
    '''ptuning v2的Trainer
    1) 虽然有past_key_values输入, 但是position_ids是从0开始的
    '''
    def __init__(self, encoder, *args, pre_seq_len=128, prefix_projection=False, **kwargs):
        super().__init__(*args, **kwargs)
        # 建立模型，加载权重
        self.encoder = encoder
        if hasattr(encoder, 'model_type'):
            self.model_type = self.encoder.model_type
        self.config = self.encoder.configs
        self.config.pre_seq_len = pre_seq_len
        self.config.prefix_projection = prefix_projection
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

    def get_past_key_values(self, token_ids):
        batch_size = token_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(token_ids.device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(torch.float16)
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.prefix_encoder.shape_3,
            self.prefix_encoder.shape_4
        )
        # b, nh, seq_len, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values
    
    def forward(self, token_ids):
        past_key_values = self.get_past_key_values(token_ids)
        logits = self.encoder([token_ids], past_key_values=past_key_values, past_key_values_length=0)
        return logits
    
    @torch.no_grad()
    def predict(self, inputs, **inputs_kwargs):
        if self.training:
            self.eval()
        token_ids = inputs[0]
        # use_states=False时候，每次都重新生成past_key_values
        # use_states=True时候，仅在第0步生成past_key_values
        if inputs_kwargs.get('past_key_values', None) is None:
            past_key_values = self.get_past_key_values(token_ids)
            inputs_kwargs['past_key_values'] = past_key_values
        inputs_kwargs['past_key_values_length'] = 0
        return self.encoder([token_ids],  **inputs_kwargs)
    