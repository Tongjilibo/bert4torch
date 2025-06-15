from torch import nn
import torch
from bert4torch.layers.position_encoding import RopePositionEncoding


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373

    :param hidden_size: 即模型最顶层输出的hidden_size
    :param heads: 在实体识别和关系提取中，分别代表着实体个数和关系个数
    :param head_size: 即每个heads的神经元个数，点积时候使用，相当于attention
    """
    def __init__(self, hidden_size:int, heads:int, head_size:int, RoPE:bool=True, use_bias:bool=True, tril_mask:bool=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RopePositionEncoding(head_size)

    def forward(self, inputs:torch.Tensor, mask:torch.Tensor=None, position_ids:torch.Tensor=None):
        ''' 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        '''
        sequence_output = self.dense(inputs)  # [..., heads*head_size*2]
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2)  # [..., heads, head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            if position_ids is None:
                seq_len = qw.shape[1]
                position_ids = torch.arange(0, seq_len, dtype=torch.float, device=qw.device).unsqueeze(0)
            # 为了seq_len维度在-2, 所以进行了转置
            qw = self.position_embedding(qw.transpose(1,-2), position_ids).transpose(1,-2)
            kw = self.position_embedding(kw.transpose(1,-2), position_ids).transpose(1,-2)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.head_size**0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """
    def __init__(self, hidden_size:int, heads:int, head_size:int, RoPE:bool=True, use_bias:bool=True, tril_mask:bool=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RopePositionEncoding(head_size)

    def forward(self, inputs:torch.Tensor, mask:torch.Tensor=None, position_ids:torch.Tensor=None):
        ''' 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        '''
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.RoPE:
            if position_ids is None:
                seq_len = qw.shape[1]
                position_ids = torch.arange(0, seq_len, dtype=torch.float, device=qw.device).unsqueeze(0)
            qw = self.position_embedding(qw, position_ids)
            kw = self.position_embedding(kw, position_ids)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2) / 2 # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits