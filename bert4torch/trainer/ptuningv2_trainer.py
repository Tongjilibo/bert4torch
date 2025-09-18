'''
ptuning v2 trainer
'''
import torch
from torch4keras.trainer import AutoTrainer, Trainer
from bert4torch.models import BaseModel
from torch import nn


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
        if config.get('num_key_value_heads') is not None:
            self.shape_3 = config.num_key_value_heads
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


class PtuningV2Model(BaseModel):
    '''ptuning v2的Model
    1) 虽然有past_key_values输入, 但是position_ids是从0开始的

    :param encoder: 预训练模型的Module
    :param pre_seq_len: int, 文本分类的类型数
    :param classifier_dropout: float, dropout比例
    :param pool_strategy: str, 选取句向量的策略, 默认为cls
    :param kwargs: dict, 其他build_transformer_model使用到的参数

    Examples
    ```python
    >>> from bert4torch.trainer import PtuningV2Model
    >>> from bert4torch.models import build_transformer_model
    >>> config_path = ''  # bert4torch_config.json路径
    >>> checkpoint_path = ''  # 模型文件夹路径
    >>> encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
    >>> model = PtuningV2Model(encoder).to('cuda')
    ```
    '''
    def __init__(self, encoder:nn.Module, *args, pre_seq_len:int=128, prefix_projection:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        # 建立模型，加载权重
        self.encoder = encoder
        if hasattr(encoder, 'model_type'):
            self.model_type = self.encoder.model_type
        self.config = self.encoder.config
        self.config.pre_seq_len = pre_seq_len
        self.config.prefix_projection = prefix_projection
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

    def get_past_key_values(self, token_ids):
        batch_size = token_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(token_ids.device)  # [btz, pre_seq_len]
        past_key_values = self.prefix_encoder(prefix_tokens).type(torch.float16)  # [btz, pre_seq_len, num_hidden_layers * hidden_size * 2]
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.prefix_encoder.shape_3,
            self.prefix_encoder.shape_4
        ) # [btz, pre_seq_len, num_hidden_layers * 2, num_attention_heads, attention_head_size]
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        past_key_values = [(v[0], v[1]) for v in past_key_values]
        # 其中每个元素的尺寸: [btz, num_attention_heads, pre_seq_len, attention_head_size]
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


class PtuningV2Trainer(AutoTrainer):
    '''PtuningV2Trainer
    :param encoder: 预训练模型的Module
    :param pre_seq_len: int, 文本分类的类型数
    :param classifier_dropout: float, dropout比例
    :param pool_strategy: str, 选取句向量的策略, 默认为cls

    Examples
    ```python
    >>> from bert4torch.trainer import PtuningV2Trainer
    >>> from bert4torch.models import build_transformer_model
    >>> config_path = ''  # bert4torch_config.json路径
    >>> checkpoint_path = ''  # 模型文件夹路径
    >>> encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
    >>> model = PtuningV2Trainer(encoder).to('cuda')
    ```
    '''
    def __init__(self, encoder:nn.Module, *args, pre_seq_len:int=128, prefix_projection:bool=False, **kwargs):
        pass

    def __new__(cls, encoder:nn.Module, *args, pre_seq_len:int=128, prefix_projection:bool=False, **kwargs) -> Trainer:
        module = PtuningV2Model(encoder, *args, pre_seq_len=pre_seq_len, prefix_projection=prefix_projection, **kwargs)
        module.to(encoder.device)
        return super().__new__(cls, module, *args, **kwargs)