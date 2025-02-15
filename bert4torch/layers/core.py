from torch import nn
import torch
import torch.nn.functional as F
from bert4torch.activations import get_activation
from bert4torch.layers.position_encoding import SinusoidalPositionEncoding
from bert4torch.snippets import torch_div, take_along_dim
from typing import Union, Literal, Optional, List, Tuple


class LayerNorm(nn.Module):
    def __init__(self, hidden_size:int, eps:float=1e-12, conditional_size:Union[bool, int]=False, weight:bool=True, bias:bool=True, 
                 norm_mode:Literal['normal', 'torch_buildin', 'rmsnorm']='normal', rmsnorm_fp32:Literal['llama-qwen', 'glm']='llama-qwen', **kwargs):
        """ layernorm层，自行实现是为了兼容conditianal layernorm，使得可以做条件文本生成、条件分类等任务

            :param hidden_size: int, layernorm的神经元个数
            :param eps: float
            :param conditional_size: int, condition layernorm的神经元个数; 详情：https://spaces.ac.cn/archives/7124
            :param weight: bool, 是否包含权重
            :param bias: bool, 是否包含偏置
            :param norm_mode: str, `normal`, `rmsnorm`, `torch_buildin`
            :param rmsnorm_fp32: str
        """
        super(LayerNorm, self).__init__()
        assert norm_mode in {'normal', 'rmsnorm', 'torch_buildin'}, f'Args norm_mode:{norm_mode} not supported'
        self.normalized_shape = (hidden_size,)
        self.norm_mode = norm_mode
        assert rmsnorm_fp32 in {'llama-qwen', 'glm'}
        self.rmsnorm_fp32 = rmsnorm_fp32
        self.eps = eps
        self.conditional_size = conditional_size
        
        # 兼容roformer_v2不包含weight
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        # 兼容t5不包含bias项, 和t5使用的RMSnorm
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None
        # 条件layernorm, 用于条件文本生成
        if conditional_size:
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, hidden_states:torch.FloatTensor, cond:Optional[torch.Tensor]=None):
        if isinstance(hidden_states, (list, tuple)):  # 兼容以前的久逻辑，后期测试后可删除
            cond = hidden_states[1] if self.conditional_size else None
            hidden_states = hidden_states[0]

        # torch自带LayerNorm
        if self.norm_mode == 'torch_buildin':
            return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)
        
        # RMSnorm: t5、大模型系列均使用
        elif self.norm_mode == 'rmsnorm':
            hidden_states_fp32 = hidden_states.float()
            variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
            if self.rmsnorm_fp32 == 'llama-qwen':
                o = (hidden_states_fp32 * torch.rsqrt(variance + self.eps)).type_as(hidden_states)  # LLAMA, QWEN
            elif self.rmsnorm_fp32 == 'glm':  # glm
                o = (hidden_states * torch.rsqrt(variance + self.eps))
        
        # 自行实现的LayerNorm
        else:
            u = hidden_states.mean(-1, keepdim=True)
            s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
            o = (hidden_states - u) / torch.sqrt(s + self.eps)

        if not hasattr(self, 'weight'):
            self.weight = 1

        if self.conditional_size and (cond is not None):
            for _ in range(len(hidden_states.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            output = (self.weight + self.dense1(cond)) * o + self.dense2(cond)
        else:
            output = self.weight * o

        if hasattr(self, 'bias') and (self.bias is not None):
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)

    def extra_repr(self) -> str:
        return f"{self.__dict__['normalized_shape']}, eps={self.__dict__['eps']}, norm_mode={self.__dict__['norm_mode']}, bias={self.bias is not None}"


class BertEmbeddings(nn.Module):
    """embeddings层
       构造word, position and token_type embeddings, 一般是token、position、segment三者embedding之和
    """
    def __init__(self, vocab_size:int, embedding_size:int, hidden_size:int, max_position:int, segment_vocab_size:int, shared_segment_embeddings:bool, 
                 dropout_rate:float, conditional_size:Union[bool, int]=False, pad_token_id:int=0, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)

        # 位置编码
        if kwargs.get('p_bias') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)
        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'MultiHeadAttention', 'deberta_v2', 'alibi'}:
            # 如果使用相对位置编码，则不声明PositionEmbeddings
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        # 层次位置编码
        self.hierarchical_position = kwargs.get('hierarchical_position')

        # segement编码
        if (segment_vocab_size > 0) and (not shared_segment_embeddings) and kwargs.get('use_segment_embedding', True):
            # use_segment_embedding用于lm, unilm场景，不使用segment_embeddings但是传入segment_ids用于计算mask
            # 一般无需设置，目前仅在guwenbert中使用
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        # emb_scale
        self.emb_scale = kwargs.get('emb_scale', 1)  # transform_xl, xlnet特有

        # LayerNorm
        self.layerNorm = LayerNorm(embedding_size, eps=kwargs.get('layer_norm_eps', 1e-12), conditional_size=conditional_size, **kwargs)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else lambda x: x

        # 如果embedding_size != hidden_size，则再有一个linear(适用于albert矩阵分解)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def apply_hierarchical_pos_embedding(self, position_index):
        """层次分解位置代码: https://spaces.ac.cn/archives/7947"""
        alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
        embeddings = self.position_embeddings.weight - alpha * self.position_embeddings.weight[:1]
        embeddings = embeddings / (1 - alpha)
        
        # 这里实现略作改动，bert4keras中是torch.arange(seq_len)[:, None]，实际使用中position_index未必是从0开始，比如padding在左侧
        btz, seqlen = position_index.shape
        position_index_reshape = position_index.flatten()[:, None]
        # 为兼容低版本pytorch没有take_along_dim
        embeddings_x = take_along_dim(embeddings,  torch_div(position_index_reshape, embeddings.size(0), rounding_mode='trunc'), dim=0)  # 兼容老版本
        embeddings_y = take_along_dim(embeddings, position_index_reshape % embeddings.size(0), dim=0)
        position_embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        return position_embeddings.reshape(btz, seqlen, -1)  # [btz, seq_len, embed_size]

    def apply_embeddings(self, token_ids:torch.Tensor, segment_ids:torch.Tensor, position_ids:torch.Tensor, 
                         additional_embs:Union[Tuple[torch.Tensor], List[torch.Tensor]]=None, **kwargs):
        '''单独拆分出来，方便下游继承和修改'''
        # word embedding
        if (not token_ids.requires_grad) and (token_ids.dtype in {torch.long, torch.int}):
            words_embeddings = self.word_embeddings(token_ids)
        else:
            words_embeddings = token_ids  # 自定义word_embedding，目前仅有VAT中使用

        # segment_embeddings
        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)  
            embeddings = words_embeddings + segment_embeddings
        elif self.shared_segment_embeddings:  # segment和word_embedding共享权重
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)  
            embeddings = words_embeddings + segment_embeddings
        else:
            embeddings = words_embeddings
        
        # position_embeddings
        if hasattr(self, 'position_embeddings') and (position_ids is not None):
            if position_ids.shape[0] == 1:  # btz维度
                position_ids = position_ids.repeat(token_ids.shape[0], 1)
            
            if (self.hierarchical_position is not None) and (position_ids.shape[1] > self.position_embeddings.weight.shape[0]):
                # 层次分解位置编码
                position_embeddings = self.apply_hierarchical_pos_embedding(position_ids)
            else:
                position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 额外的embedding，如词性等
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb
        return embeddings

    def forward(self, token_ids:torch.Tensor=None, segment_ids:torch.Tensor=None, position_ids:torch.Tensor=None, conditional_emb:Optional[torch.Tensor]=None, 
                additional_embs:Union[Tuple[torch.Tensor], List[torch.Tensor]]=None, attention_mask:torch.Tensor=None, **kwargs):
        embeddings = self.apply_embeddings(token_ids, segment_ids, position_ids, additional_embs, **kwargs)

        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale  # transform_xl, xlnet特有

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm(embeddings, conditional_emb)
        
        if attention_mask is not None:
            embeddings *= attention_mask[:, 0, 0, :, None]

        if hasattr(self, 'dropout'):
            embeddings = self.dropout(embeddings)

        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int, dropout_rate:float=0.5, hidden_act:str='gelu', is_dropout:bool=False, bias:bool=True, **kwargs):
        # 原生的tf版本的bert在激活函数后，没有添加dropout层，但是在google AI的bert-pytorch开源项目中，多了一层dropout；
        # 并且在pytorch官方的TransformerEncoderLayer的实现中，也有一层dropout层，就像这样：self.linear2(self.dropout(self.activation(self.linear1(src))))；
        # 这样不统一做法的原因不得而知，不过有没有这一层，差别可能不会很大；

        # 为了适配是否dropout，用is_dropout，dropout_rate两个参数控制；如果是实现原始的transformer，直接使用默认参数即可；如果是实现bert，则is_dropout为False，此时的dropout_rate参数并不会使用.
        super(PositionWiseFeedForward, self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = get_activation(hidden_act)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size*2 if hidden_act=='swiglu' else intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch size, seq len, hidden_size)
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))

        # x shape: (batch size, seq len, intermediate_size)
        x = self.outputDense(x)

        # x shape: (batch size, seq len, hidden_size)
        return x


class LlamaFeedForward(nn.Module):
    '''FeedForward和Bert的不一致，Bert只有两个全连接, LLaMA和Qwen使用'''
    def __init__(self, dim: int, intermediate_size: int, hidden_act='silu', bias=False, **kwargs):
        super().__init__()
        self.intermediateDense = nn.Linear(dim, intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, dim, bias=bias)
        self.intermediateDense2 = nn.Linear(dim, intermediate_size, bias=bias)
        self.intermediate_act_fn = get_activation(hidden_act)

    def forward(self, x):
        return self.outputDense(self.intermediate_act_fn(self.intermediateDense(x)) * self.intermediateDense2(x))


class T5PositionWiseFeedForward(PositionWiseFeedForward):
    '''参考transformer包: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py'''
    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(hidden_size, intermediate_size, **kwargs)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.intermediateDense1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # x shape: (batch size, seq len, hidden_size)
        x_gelu = self.intermediate_act_fn(self.intermediateDense(x))
        x_linear = self.intermediateDense1(x)
        x = x_gelu * x_linear
        if self.is_dropout:
            x = self.dropout(x)

        # x shape: (batch size, seq len, intermediate_size)
        x = self.outputDense(x)

        # x shape: (batch size, seq len, hidden_size)
        return x