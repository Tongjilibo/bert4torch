from torch import nn
import torch
import torch.nn.functional as F
from typing import Union, Dict, Type, Any
from bert4torch.snippets import create_registrar



LAYER_NORM_MAP: Dict[str, Type[nn.Module]] = {}
register_layer_norm = create_registrar(LAYER_NORM_MAP)


class CLS_LAYER_NORM:
    """封装后的LayerNorm映射类，支持直接传入kwargs获取对应类型"""
    def __init__(self, base_registry: Dict[str, Type[nn.Module]]):
        self.base_registry = base_registry  # 关联原始的LAYER_NORM注册表

    def __getitem__(self, kwargs: Dict[str, Any]) -> Type[nn.Module]:
        if kwargs.get('conditional_size') is not None:
            layer_norm_mode = 'conditional_layer_norm'
        else:
            layer_norm_mode = kwargs.get('layer_norm_mode', 'default')
        return LAYER_NORM_MAP[layer_norm_mode]

LAYER_NORM = CLS_LAYER_NORM(LAYER_NORM_MAP)


@register_layer_norm(name='default')
class LayerNorm(nn.Module):
    def __init__(self, hidden_size:int, layer_norm_eps:float=1e-12, **kwargs):
        """ layernorm层，自行实现是为了兼容conditianal layernorm，使得可以做条件文本生成、条件分类等任务
            :param hidden_size: int, layernorm的神经元个数
            :param eps: float
            :param bias: bool, 是否包含偏置
        """
        super(LayerNorm, self).__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = layer_norm_eps

        self.weight = nn.Parameter(torch.ones(hidden_size))
        if kwargs.get('norm_bias') or kwargs.get('use_bias', True):
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.bias = None

    def _normalize(self, hidden_states:torch.FloatTensor, **kwargs):
        '''标准化'''
        u = hidden_states.mean(-1, keepdim=True)
        s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
        o = (hidden_states - u) / torch.sqrt(s + self.eps)
        return o
    def forward(self, hidden_states:torch.FloatTensor, *args, **kwargs):
        o = self._normalize(hidden_states, **kwargs)
        output = self.weight * o

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)

    def __repr__(self) -> str:
        return f"{self.__dict__['normalized_shape']}, eps={self.__dict__['eps']}, norm_mode={self.__dict__['norm_mode']}, bias={self.bias is not None}"


@register_layer_norm(name='conditional_layer_norm')
class ConditionalLayerNorm(LayerNorm):
    '''
    :param conditional_size: int, condition layernorm的神经元个数; 详情：https://spaces.ac.cn/archives/7124
    '''
    def __init__(self, hidden_size, *args, conditional_size:Union[bool, int]=False, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        # 条件layernorm, 用于条件文本生成
        self.conditional_size = conditional_size
        if conditional_size:
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)
    
    def forward(self, hidden_states, *args, conditional_emb = None, **kwargs):
        if conditional_emb is None and len(args) > 0:  # 兼容以前的久逻辑，后期测试后可删除
            conditional_emb = args[0] if self.conditional_size else None

        o = self._normalize(hidden_states, **kwargs)
        
        if self.conditional_size and (conditional_emb is not None):
            for _ in range(len(hidden_states.shape) - len(conditional_emb.shape)):
                conditional_emb = conditional_emb.unsqueeze(dim=1)
            output = (self.weight + self.dense1(conditional_emb)) * o + self.dense2(conditional_emb)

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)
    

@register_layer_norm(name='torch_buildin')
class TorchBuildInLayerNorm(LayerNorm):
    def forward(self, hidden_states, *args, **kwargs):
        return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)


@register_layer_norm
@register_layer_norm(name='roformer_v2')
@register_layer_norm(name='gau_alpha')
class RoformerV2LayerNorm(LayerNorm):
    '''RoFormerV2和GAU_alpha没有weight'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.weight
    
    def forward(self, hidden_states:torch.FloatTensor, *args, **kwargs):
        output = self._normalize(hidden_states, **kwargs)

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)


@register_layer_norm(name='rmsnorm')
class RMSNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        kwargs['use_bias'] = False
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        # RMSnorm: t5、大模型系列均使用
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        o = (hidden_states_fp32 * torch.rsqrt(variance + self.eps)).type_as(hidden_states)  # LLAMA, QWEN
        output = self.weight * o

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)


@register_layer_norm(name='glm_rmsnorm')
class GlmRMSNorm(RMSNorm):
    def forward(self, hidden_states, *args, **kwargs):
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        o = (hidden_states * torch.rsqrt(variance + self.eps))
        output = self.weight * o

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)
