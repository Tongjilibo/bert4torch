from torch import nn
import torch
import math
import torch.nn.functional as F
from typing import Union, List


def get_sinusoid_encoding_table(n_position, d_hid, base=10000.0, ntk_alpha=1.0, rope_ratio=1.0, padding_idx=None):
    ''' sinusoid编码
        
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :param padding_idx: padding的token_ids
        :param ntk_alpha: int, 要扩展的倍数
        :param rope_ratio: int, chatglm中32k的插值
        :return: [seq_len, d_hid]
    '''
    if ntk_alpha != 1:
        base = base * ntk_alpha ** (d_hid / (d_hid-2))
    
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(base) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    if rope_ratio != 0:
        position = position / rope_ratio
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

    # 第二种实现
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids


class RelativePositionsEncodingDebertaV2(nn.Module):
    """deberta用的相对位置编码
    来自论文：https://arxiv.org/abs/2006.03654
    """
    def __init__(self, qlen, klen, position_buckets, max_position):
        super(RelativePositionsEncodingDebertaV2, self).__init__()
        q_ids = torch.arange(0, qlen)
        k_ids = torch.arange(0, klen)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        if position_buckets > 0 and max_position > 0:
            rel_pos_ids = self.make_log_bucket_position(rel_pos_ids, position_buckets, max_position)
        rel_pos_ids = rel_pos_ids.to(torch.long)
        rel_pos_ids = rel_pos_ids[:qlen, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        self.register_buffer('relative_position', rel_pos_ids)

    @staticmethod
    def make_log_bucket_position(relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid),
            torch.tensor(mid - 1).type_as(relative_pos),
            torch.abs(relative_pos),
        )
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
        return bucket_pos

    def forward(self, qlen, klen):
        return self.relative_position[:, :qlen, :klen]


class RelativePositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(RelativePositionsEncoding, self).__init__()
        # 生成相对位置矩阵
        vocab_size = max_relative_position * 2 + 1
        distance_mat = torch.arange(klen)[None, :] - torch.arange(qlen)[:, None]  # 列数-行数, [query_len, key_len]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        # sinusoid_encoding编码的位置矩阵
        embeddings_table = get_sinusoid_encoding_table(vocab_size, embedding_size)

        # 实现方式1
        # flat_relative_positions_matrix = final_mat.view(-1)
        # one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix, num_classes=vocab_size).float()
        # position_embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
        # my_shape = list(final_mat.size())
        # my_shape.append(embedding_size)
        # position_embeddings = position_embeddings.view(my_shape)

        # 实现方式2
        # position_embeddings = take_along_dim(embeddings_table, final_mat.flatten().unsqueeze(1), dim=0)
        # position_embeddings = position_embeddings.reshape(*final_mat.shape, embeddings_table.shape[-1])  # [seq_len, seq_len, hdsz]
        # self.register_buffer('position_embeddings', position_embeddings)
        
        # 实现方式3
        position_embeddings = nn.Embedding.from_pretrained(embeddings_table, freeze=True)(final_mat)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, qlen, klen):
        return self.position_embeddings[:qlen, :klen, :]


class RelativePositionsEncodingT5(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(RelativePositionsEncodingT5, self).__init__()
        # 生成相对位置矩阵
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        relative_position = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not is_decoder,
            num_buckets=relative_attention_num_buckets,
        )
        self.register_buffer('relative_position', relative_position)

    def forward(self, qlen, klen):
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        '''直接来源于transformer
        '''
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret


class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=True) 
    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265

    :param embedding_size: embedding的大小
    :param rope_rank: 排序的方式，目前支持'adjacent', 'updown'两种
    :param ntk_alpha: ntk外推的alpha
    """
    def __init__(self, embedding_size, max_seq_len_cache=2048, rope_rank='adjacent', ntk_alpha=1.0, rope_ratio=1.0, rope_theta=10000.0, **kwargs):
        super(RoPEPositionEncoding, self).__init__()
        self.max_seq_len_cache = -1
        self.embedding_size = embedding_size
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列, 目前只在chatglm中看到updown排列
        assert rope_rank in {'adjacent', 'updown', 'rotate_half'}, "rank kwarg only support 'adjacent/updown/rotate_half' "
        self.rope_rank = rope_rank
        self.ntk_alpha = ntk_alpha  # ntk外推
        self.rope_ratio = rope_ratio  # chatglm中32k的插值
        self.rope_theta = rope_theta
        self.max_seq_len_cache = max_seq_len_cache
        self._set_cos_sin_cache(max_seq_len_cache)

    def reset_ntk_alpha(self, ntk_alpha):
        if ntk_alpha != self.ntk_alpha:
            self.ntk_alpha = ntk_alpha
            self.max_seq_len_cache = -1
        
    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        self.max_seq_len_cache = seq_len
        position_embeddings = get_sinusoid_encoding_table(seq_len, self.embedding_size, base=self.rope_theta,
                                                          ntk_alpha=self.ntk_alpha, rope_ratio=self.rope_ratio)  # [seq_len, hdsz]

        if self.rope_rank == 'adjacent':
            # 相邻的两位是相同的，和官方博客上一致，如cos_position是[cos(mθ0), cos(mθ0), cos(mθ1), cos(mθ1), ...] 
            cos_cache = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
            sin_cache = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
        elif self.rope_rank in {'updown', 'rotate_half'}:  # 目前chatglm和llama系列有部分使用
            # 整片的上下分布，和官方博客上不一致，如cos_position是[cos(mθ0), cos(mθ1), ..., cos(mθ(d/2-1)), cos(mθ0), cos(mθ1), ..., cos(mθ(d/2-1))] 
            cos_cache = position_embeddings[:, 1::2].repeat(1,2)  # [seq_len, hdsz]
            sin_cache = position_embeddings[:, ::2].repeat(1,2)  # [seq_len, hdsz]

        self._register_buffer(cos_cache, sin_cache, device, dtype)

    def _register_buffer(self, cos_cache, sin_cache, device=None, dtype=None):
        if device is not None:
            cos_cache, sin_cache = cos_cache.to(device), sin_cache.to(device)
        if dtype is not None:
            cos_cache, sin_cache = cos_cache.to(dtype), sin_cache.to(dtype)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def rotate_and_compute(self, x, cos, sin):
        # MultiHeadAttentionLayer中x是[btz, n_heads, seq_len, head_size]
        # GlobalPointer中*转置*后x是[btz, n_heads, seq_len, head_size]
        # EfficientGlobalPointer中x是[btz, seq_len, head_size]
        if self.rope_rank == 'adjacent':
            x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        elif self.rope_rank in {'updown', 'rotate_half'}:
            # 其实就是rotate_half，注意cat和stack+reshape是结果不同的
            x2 = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return x * cos + x2 * sin

    def forward(self, qk:Union[torch.Tensor, List], position_ids=None, seq_len=None, seq_dim=-2):
        '''修改了原有的q和k重复走一遍embedding，实现加速'''
        if isinstance(qk, list):
            device, dtype = qk[0].device, qk[0].dtype
        else:
            device, dtype = qk.device, qk.dtype

        # 超过缓存长度
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max() + 1
            elif isinstance(qk, list):
                seq_len = qk[0].shape[seq_dim]
            elif isinstance(qk, torch.Tensor):
                seq_len = qk.shape[seq_dim]
        
        if seq_len > self.max_seq_len_cache:
            self._set_cos_sin_cache(seq_len, device, dtype)
        if (self.cos_cache.dtype != dtype) or (self.cos_cache.device != device):
            self._register_buffer(self.cos_cache, self.sin_cache, device, dtype)
        
        # 传入position_ids来获取cos和sin, 主要是在use_states时候能直接取到对应位置的编码
        if position_ids is not None:
            # position_ids: [btz, seq_len]
            cos = F.embedding(position_ids, self.cos_cache)  # [btz, seq_len, hdsz]
            sin = F.embedding(position_ids, self.sin_cache)
        else:
            cos = self.cos_cache[:seq_len]  # [seq_len, hdsz]
            sin = self.sin_cache[:seq_len]

        if cos.dim() < qk[0].dim() if isinstance(qk, list) else qk.dim():
            cos = cos.unsqueeze(seq_dim-1)
            sin = sin.unsqueeze(seq_dim-1)

        if isinstance(qk, list):
            return [self.rotate_and_compute(x, cos, sin) for x in qk]
        else:
            return self.rotate_and_compute(qk, cos, sin)


class XlnetPositionsEncoding(nn.Module):
    '''Xlnet, transformer_xl使用的相对位置编码
       和SinusoidalPositionEncoding区别是一个是间隔排列, 一个是前后排列
    '''
    def __init__(self, embedding_size):
        super().__init__()
        self.demb = embedding_size
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_size, 2.0) / embedding_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class ALiBiPositionsEncoding(nn.Module):
    '''ALiBi: Attention with Linear Biases
       https://github.com/ofirpress/attention_with_linear_biases
    '''
    def __init__(self, n_head, **kwargs) -> None:
        super().__init__()
        self.n_head = n_head
        self.max_cache_pos = -1
    
    def _get_interleave(self, n):
        def _get_interleave_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2) + \
                self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def _gen_alibi_mask(self, seq_len):
        slopes = torch.Tensor(self._get_interleave(self.n_head))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(self.n_head, -1, -1)
        alibi = alibi.view(self.n_head, 1, seq_len)
        return alibi
    
    def forward(self, key_layer):
        '''
        key_layer: [btz, n_head, q_len, hdsz]
        '''
        seq_length_with_past = key_layer.shape[2]
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.future_mask = self._gen_alibi_mask(seq_length_with_past).to(key_layer)
        
        mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past] 
        return mask.unsqueeze(0)