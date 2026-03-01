from torch import nn
import torch
from typing import Union, Optional, List, Tuple, Dict, Type
from .position_encoding import SinusoidalPositionEncoding
from .layer_norm import LayerNorm, LAYER_NORM
from bert4torch.snippets import torch_div, take_along_dim, create_registrar


EMBEDDING_MAP : Dict[str, Type[nn.Module]] = {}
register_embedding = create_registrar(EMBEDDING_MAP)


@register_embedding
@register_embedding(name='default')
class BertEmbeddings(nn.Module):
    """embeddings层
       构造word, position and token_type embeddings, 一般是token、position、segment三者embedding之和
    """
    def __init__(self, vocab_size:int, embedding_size:int, hidden_size:int, max_position_embeddings:int, segment_vocab_size:int, shared_segment_embeddings:bool, 
                 dropout_rate:float, pad_token_id:int=0, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)

        # 位置编码
        if kwargs.get('pos_emb_type') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position_embeddings, embedding_size)
        elif kwargs.get('pos_emb_type') in {'rotary', 'typical_relative', 't5_relative', 'MultiHeadAttention', 'deberta_v2', 'alibi'}:
            # 如果使用相对位置编码，则不声明PositionEmbeddings
            pass
        elif max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)

        # segement编码
        if (segment_vocab_size > 0) and (not shared_segment_embeddings) and kwargs.get('use_segment_embedding', True):
            # use_segment_embedding用于lm, unilm场景，不使用segment_embeddings但是传入segment_ids用于计算mask
            # 一般无需设置，目前仅在guwenbert中使用
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        # emb_scale
        self.emb_scale = kwargs.get('emb_scale', 1)  # transform_xl, xlnet特有

        # LayerNorm
        self.layerNorm = LAYER_NORM[kwargs](embedding_size, **kwargs)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else lambda x: x

        # 如果embedding_size != hidden_size，则再有一个linear(适用于albert矩阵分解)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def _apply_word_embeddings(self, token_ids, **kwargs):
        '''word embedding'''
        if (not token_ids.requires_grad) and (token_ids.dtype in {torch.long, torch.int}):
            word_embeddings = self.word_embeddings(token_ids)
        else:
            # VL大模型 / 自定义word_embedding，目前仅有VAT中使用
            word_embeddings = token_ids
        return word_embeddings

    def _apply_segment_embeddings(self, token_ids, segment_ids, **kwargs):
        '''segment embedding'''
        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)  
        elif self.shared_segment_embeddings:  # segment和word_embedding共享权重
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)  
        else:
            segment_embeddings = 0
        return segment_embeddings
    
    def _apply_position_embeddings(self, token_ids, position_ids, **kwargs):
        '''position embedding'''
        position_embeddings = 0  # 默认值
        if hasattr(self, 'position_embeddings') and (position_ids is not None):
            if position_ids.shape[0] == 1:  # btz维度
                position_ids = position_ids.repeat(token_ids.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

    def _post_process(self, embeddings, conditional_emb, attention_mask, **kwargs):
        '''post process'''
        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale  # transform_xl, xlnet特有

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm(embeddings, conditional_emb=conditional_emb)
        
        if attention_mask is not None:
            embeddings *= attention_mask[:, 0, 0, :, None]

        if hasattr(self, 'dropout'):
            embeddings = self.dropout(embeddings)

        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings
    
    def forward(self, token_ids:torch.Tensor=None, segment_ids:torch.Tensor=None, position_ids:torch.Tensor=None, conditional_emb:Optional[torch.Tensor]=None, 
                additional_embs:Union[Tuple[torch.Tensor], List[torch.Tensor]]=None, attention_mask:torch.Tensor=None, **kwargs):
        # word embedding
        word_embeddings = self._apply_word_embeddings(token_ids, **kwargs)

        # segment_embeddings
        segment_embeddings = self._apply_segment_embeddings(token_ids, segment_ids, **kwargs)
        
        # position_embeddings
        position_embeddings = self._apply_position_embeddings(token_ids, position_ids, **kwargs)

        # 额外的embedding，如词性等
        embeddings = word_embeddings + segment_embeddings + position_embeddings
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb
    
        embeddings = self._post_process(embeddings, conditional_emb, attention_mask, **kwargs)
        return embeddings


@register_embedding
class HierarchicalPositionEmbeddings(BertEmbeddings):
    """层次分解位置代码: https://spaces.ac.cn/archives/7947"""
    def __init__(self, *args, hierarchical_position_alpha:float=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchical_position_alpha = hierarchical_position_alpha

    def _apply_hierarchical_pos_embedding(self, position_ids):
        embeddings = self.position_embeddings.weight - self.hierarchical_position_alpha * self.position_embeddings.weight[:1]
        embeddings = embeddings / (1 - self.hierarchical_position_alpha)
        
        # 这里实现略作改动，bert4keras中是torch.arange(seq_len)[:, None]，实际使用中position_index未必是从0开始，比如padding在左侧
        btz, seqlen = position_ids.shape
        position_index_reshape = position_ids.flatten()[:, None]
        # 为兼容低版本pytorch没有take_along_dim
        embeddings_x = take_along_dim(embeddings,  torch_div(position_index_reshape, embeddings.size(0), rounding_mode='trunc'), dim=0)  # 兼容老版本
        embeddings_y = take_along_dim(embeddings, position_index_reshape % embeddings.size(0), dim=0)
        position_embeddings = self.hierarchical_position_alpha * embeddings_x + (1 - self.hierarchical_position_alpha) * embeddings_y
        return position_embeddings.reshape(btz, seqlen, -1)  # [btz, seq_len, embed_size]

    def _apply_position_embeddings(self, token_ids, position_ids, **kwargs):
        '''position embedding'''
        position_embeddings = 0  # 默认值
        if hasattr(self, 'position_embeddings') and (position_ids is not None):
            if position_ids.shape[0] == 1:  # btz维度
                position_ids = position_ids.repeat(token_ids.shape[0], 1)
            
            if position_ids.shape[1] > self.position_embeddings.weight.shape[0]:
                # 层次分解位置编码
                position_embeddings = self._apply_hierarchical_pos_embedding(position_ids)
            else:
                position_embeddings = self.position_embeddings(position_ids)
            
        return position_embeddings


@register_embedding
class ErnieEmbeddings(BertEmbeddings):
    def __init__(self, vocab_size, embedding_size, *args, **kwargs):
        super().__init__(vocab_size, embedding_size, *args, **kwargs)
        self.use_task_id = kwargs.get('use_task_id')

        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), embedding_size)
    
    def _apply_segment_embeddings(self, token_ids, segment_ids, **kwargs):
        '''把task_type_embeddings混到到segment_embeddings中'''
        segment_embeddings = super()._apply_segment_embeddings(token_ids, segment_ids, **kwargs)

        task_type_ids = kwargs.get('task_type_ids')
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(token_ids.shape, dtype=torch.long, device=segment_embeddings.device)
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            segment_embeddings += task_type_embeddings
        return segment_embeddings


