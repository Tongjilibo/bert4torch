from torch import nn
import torch
import torch.nn.functional as F
from typing import Union, Literal, Optional


class LayerNorm(nn.Module):
    def __init__(self, hidden_size:int, layer_norm_eps:float=1e-12, conditional_size:Union[bool, int]=False,  
                 layer_norm_mode:Literal['normal', 'torch_buildin', 'rmsnorm']='normal', 
                 rmsnorm_fp32:Literal['llama-qwen', 'glm']='llama-qwen', **kwargs):
        """ layernorm层，自行实现是为了兼容conditianal layernorm，使得可以做条件文本生成、条件分类等任务

            :param hidden_size: int, layernorm的神经元个数
            :param eps: float
            :param conditional_size: int, condition layernorm的神经元个数; 详情：https://spaces.ac.cn/archives/7124
            :param bias: bool, 是否包含偏置
            :param norm_mode: str, `normal`, `rmsnorm`, `torch_buildin`
            :param rmsnorm_fp32: str
        """
        super(LayerNorm, self).__init__()
        assert layer_norm_mode in {'normal', 'rmsnorm', 'torch_buildin'}, f'Args norm_mode:{layer_norm_mode} not supported'
        self.normalized_shape = (hidden_size,)
        self.norm_mode = layer_norm_mode
        assert rmsnorm_fp32 in {'llama-qwen', 'glm'}
        self.rmsnorm_fp32 = rmsnorm_fp32
        self.eps = layer_norm_eps
        self.conditional_size = conditional_size

        # RoFormerV2和GAU_alpha没有weight
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # 兼容t5不包含bias项, 大模型的RMSnorm
        use_bias = kwargs.get('norm_bias', kwargs.get('use_bias', True))
        if not use_bias or self.norm_mode == 'rmsnorm':
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        
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

        if self.norm_mode == 'torch_buildin':
            # torch自带LayerNorm
            return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.norm_mode == 'rmsnorm':
            # RMSnorm: t5、大模型系列均使用
            hidden_states_fp32 = hidden_states.float()
            variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
            if self.rmsnorm_fp32 == 'llama-qwen':
                o = (hidden_states_fp32 * torch.rsqrt(variance + self.eps)).type_as(hidden_states)  # LLAMA, QWEN
            elif self.rmsnorm_fp32 == 'glm':  # glm
                o = (hidden_states * torch.rsqrt(variance + self.eps))
        else:
            # 自行实现的LayerNorm
            u = hidden_states.mean(-1, keepdim=True)
            s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
            o = (hidden_states - u) / torch.sqrt(s + self.eps)

        if not hasattr(self, 'weight'):
            output = o
        elif self.conditional_size and (cond is not None):
            for _ in range(len(hidden_states.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            output = (self.weight + self.dense1(cond)) * o + self.dense2(cond)
        else:
            output = self.weight * o

        if getattr(self, 'bias', None) is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)

    def extra_repr(self) -> str:
        return f"{self.__dict__['normalized_shape']}, eps={self.__dict__['eps']}, norm_mode={self.__dict__['norm_mode']}, bias={self.bias is not None}"

