''' Activation激活函数
从transformer中移植过来的activation, 原来的bert4keras并没有
'''

import math
import torch
from torch import nn
from packaging import version
from bert4torch.snippets import create_registrar


ACT2FN = {
    "relu": nn.functional.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softmax": nn.Softmax(dim=-1),
    "gelu_pytorch_tanh": lambda input: nn.functional.gelu(input, approximate="tanh")
}
register_act = create_registrar(ACT2FN)


@register_act(name='gelu')
def gelu(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    if version.parse(torch.__version__) < version.parse("1.4"):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    else:
        return nn.functional.gelu(x)


@register_act(name='gelu_new')
@register_act(name='_gelu_new')
def _gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@register_act(name='gelu_fast')
def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


@register_act(name='quick_gelu')
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@register_act(name='silu')
@register_act(name='swish')
def silu(x):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """
    if version.parse(torch.__version__) < version.parse("1.7"):
        return x * torch.sigmoid(x)
    else:
        return nn.functional.silu(x)


@register_act(name='mish')
def mish(x):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    if version.parse(torch.__version__) < version.parse("1.9"):
        return x * torch.tanh(nn.functional.softplus(x))
    else:
        return nn.functional.mish(x)


@register_act(name='linear')
def linear_act(x):
    return x


@register_act(name='swiglu')
def swiglu(x, dim=-1):
    x = torch.chunk(x, 2, dim=dim)
    return silu(x[0]) * x[1]


def get_activation(activation_string):
    '''根据activation_string返回对应的激活函数

    :param activation_string: str, 传入的激活函数名
    :return: Any
    '''
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
