from torch import nn
import torch
import math
import torch.nn.functional as F
from typing import Union, List, Literal, Optional


def get_sinusoid_encoding_table(n_position:int, d_hid:int, base:float=10000.0, padding_idx:Optional[int]=None):
    ''' sinusoid编码
        
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :param padding_idx: padding的token_ids
        :return: [seq_len, d_hid]
    '''   
    inv_freq = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(base) / d_hid))    
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * inv_freq)
    embeddings_table[:, 1::2] = torch.cos(position * inv_freq)
    return embeddings_table

    # 第二种实现
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids


class DebertaV2PositionsEncoding(nn.Module):
    """deberta用的相对位置编码
    来自论文：https://arxiv.org/abs/2006.03654
    """
    def __init__(self, qlen, klen, position_buckets, max_position):
        super(DebertaV2PositionsEncoding, self).__init__()
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


class NezhaPositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(NezhaPositionsEncoding, self).__init__()
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


class T5PositionsEncoding(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(T5PositionsEncoding, self).__init__()
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


class RopePositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265

    :param embedding_size: embedding的大小
    :param rope_rank: 排序的方式，目前支持'adjacent', 'updown/rotate_half'
    :param ntk_alpha: ntk外推的alpha
    :param scaling_factor: 对position_ids进行缩放的尺度参数
    :param rope_theta: rope中使用的base大小
    """
    def __init__(self, 
                 embedding_size: int, 
                 max_position: int=2048, 
                 rope_rank: Literal['adjacent', 'updown', 'rotate_half']='adjacent',
                 scaling_factor: float=1.0, 
                 rope_theta: float=10000.0, 
                 device = None,
                 **kwargs):
        super(RopePositionEncoding, self).__init__()
        self.embedding_size = embedding_size
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列, 目前只在chatglm中看到updown排列
        self.rope_rank = rope_rank or 'adjacent'
        assert self.rope_rank in {'adjacent', 'updown', 'rotate_half'}, "rank kwarg only support 'adjacent/updown/rotate_half' "
        self.ntk_alpha = 1.0  # ntk外推
        self.scaling_factor = scaling_factor  # chatglm中32k的插值
        self.rope_theta = rope_theta or 10000.0
        self.max_position = max_position  # 原始支持的最大长度

        # 有三种方案
        # 1. 仅register_buffer inv_freq, cos和sin都在forward现算，优点省显存，缺点降低forward速度, transformer中最新逻辑
        # 2. 按照最大长度register_buffer cos和sin，优点forward速度快，缺点费显存
        # 3. 按照给定的最小长度register_buffer cos和sin，优点forward速度折中（>最小长度需现算），缺点显存折中
        self.max_seq_len_cached = kwargs.get('max_seq_len_cached', max_position)  # 推理过程中遇到的最大长度max(seq_len, max_position)
        self.sin_cos_cached = kwargs.get('sin_cos_cached', False)
        self._set_inv_freq_cache(self.max_seq_len_cached, device=device)  # 这里没有直接设置到max_position，因为容易占显存
        self._set_cos_sin_cache(self.max_seq_len_cached, device or 'cpu', dtype=torch.get_default_dtype())
    
    def _set_inv_freq_cache(self, seq_len, device=None):
        '''计算inv_freq并且缓存，永远是float32的'''
        base = self.rope_theta
        if (self.ntk_alpha is not None) and (self.ntk_alpha != 1):
            base = base * self.ntk_alpha ** (self.embedding_size / (self.embedding_size-2))
        
        inv_freq = torch.exp(torch.arange(0, self.embedding_size, 2).float() * (-math.log(base) / self.embedding_size)).to(device)
        if (self.scaling_factor is not None) and (self.scaling_factor != 1):
            inv_freq = inv_freq / self.scaling_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        '''缓存cos和sin的值，适用于小尺寸模型，可以把cos和sin计算好缓存起来，好处是forward更快，缺点是多占显存'''
        if self.sin_cos_cached:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
            cos, sin = self._compute_cos_sin(self.inv_freq[None, :, None], position_ids[None, :], device, dtype)
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)

    def compute_cos_sin(self, qk:Union[torch.Tensor, List[torch.Tensor]], position_ids:torch.Tensor):
        '''计算cos和sin
        param position_ids: [..., btz, seq_len]
        '''
        if self.sin_cos_cached and hasattr(self, 'cos_cached') and hasattr(self, 'sin_cached'):
            # position_ids [btz, seq_len], cos_cached [btz, max_seq_len, emb_size]
            cos = torch.stack([F.embedding(i[0], i[1]) for i in zip(position_ids, self.cos_cached)])
            sin = torch.stack([F.embedding(i[0], i[1]) for i in zip(position_ids, self.sin_cached)])
            return cos, sin

        q:torch.Tensor = qk[0] if isinstance(qk, list) else qk
        device_type = q.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        self.inv_freq: torch.Tensor

        # 兼容position_ids传入不同的维度
        pos_dim = position_ids.dim()
        if pos_dim == 2:  # 一般的rope
            inv_freq_expanded = self.inv_freq[None, :, None]
        elif pos_dim == 3:  # qwen2_vl
            inv_freq_expanded = self.inv_freq[None, None, :, None]
        else:  # 通用的
            inv_freq_expanded:torch.Tensor = self.inv_freq
            for i in range(pos_dim+1):
                if i != pos_dim-1:
                    inv_freq_expanded = inv_freq_expanded.unsqueeze(i)
        
        cos, sin = self._compute_cos_sin(inv_freq_expanded, position_ids, device_type, q.dtype)
        return cos, sin

    def _compute_cos_sin(self, inv_freq_expanded:torch.Tensor, position_ids:torch.Tensor, device_type:str, dtype:str):
        '''拆分出来，方便compute_cos_sin和_set_cos_sin_cache调用'''
        inv_freq_expanded = inv_freq_expanded.float().expand(*position_ids.shape[:-1], -1, 1)
        position_ids_expanded = position_ids.unsqueeze(-2).float()

        with torch.autocast(device_type=device_type, enabled=False):
            emb = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(-2, -1)  # btz, seq_len, hdsz
            cos:torch.Tensor = emb.cos()
            sin:torch.Tensor = emb.sin()

        if self.rope_rank == 'adjacent':
            # 相邻的两位是相同的，和官方博客上一致，如cos_position是[cos(mθ0), cos(mθ0), cos(mθ1), cos(mθ1), ...] 
            cos = cos.repeat_interleave(2, dim=-1)  # [..., seq_len, hdsz]
            sin = sin.repeat_interleave(2, dim=-1)  # [..., seq_len, hdsz]
        elif self.rope_rank in {'updown', 'rotate_half'}:  # 目前chatglm和llama系列有部分使用
            # 整片的上下分布，和官方博客上不一致，如cos_position是[cos(mθ0), cos(mθ1), ..., cos(mθ(d/2-1)), cos(mθ0), cos(mθ1), ..., cos(mθ(d/2-1))] 
            cos = torch.cat((cos, cos), dim=-1)  # [..., seq_len, hdsz]
            sin = torch.cat((sin, sin), dim=-1)  # [..., seq_len, hdsz]

        return cos.to(dtype=dtype), sin.to(dtype=dtype)
    
    def rotate_and_compute(self, x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor, position_ids:torch.Tensor, unsqueeze_dim:int=1):
        # MultiHeadAttention中x是[btz, n_heads, seq_len, head_size]
        # GlobalPointer中*转置*后x是[btz, n_heads, seq_len, head_size]
        # EfficientGlobalPointer中x是[btz, seq_len, head_size]
        if self.rope_rank == 'adjacent':
            x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        elif self.rope_rank in {'updown', 'rotate_half'}:
            # 其实就是rotate_half，注意cat和stack+reshape是结果不同的
            x2 = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        if cos.dim() < x.dim():
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
        return x * cos + x2 * sin

    def forward(self, qk:Union[torch.Tensor, List[torch.Tensor]], position_ids:torch.Tensor):
        '''修改了原有的q和k重复走一遍embedding，实现加速'''
        if isinstance(qk, list):
            device, dtype = qk[0].device, qk[0].dtype
        else:
            device, dtype = qk.device, qk.dtype

        # 超过缓存长度则重新设置cache
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            self._set_inv_freq_cache(seq_len, device)
            self._set_cos_sin_cache(seq_len, device, dtype)
        
        # 计算cos和sin
        cos, sin = self.compute_cos_sin(qk, position_ids)

        if isinstance(qk, list):
            return [self.rotate_and_compute(x, cos, sin, position_ids) for x in qk]
        else:
            return self.rotate_and_compute(qk, cos, sin, position_ids)


class RopeLinearScalingPositionEncoding(RopePositionEncoding):
    '''使用linear scaling的rope, scaling_factor != 1的时候生效'''
    pass


class RopeGlmPositionEncoding(RopePositionEncoding):
    '''GLM对应的rope编码'''
    def __init__(self, 
                 embedding_size: int, 
                 max_position: int = 2048, 
                 rope_rank: Literal['adjacent', 'updown', 'rotate_half']='adjacent', 
                 scaling_factor: float = 1, 
                 rope_theta: float = 10000, 
                 device=None, 
                 **kwargs):
        # glm的embedding_size不同
        super().__init__(embedding_size // 2, max_position, rope_rank, scaling_factor, rope_theta, device, **kwargs)
    
    @torch.no_grad()
    def forward(self, qk:Union[torch.Tensor, List[torch.Tensor]], position_ids:torch.Tensor=None):
        query_states, key_states = qk
        
        q1, q2 = query_states.chunk(2, dim=(query_states.ndim - 1))
        k1, k2 = key_states.chunk(2, dim=(key_states.ndim - 1))
        if len(position_ids.shape) == 3:
            q1, k1 = super().forward([q1, k1], position_ids[:, 0, :])
            q2, k2 = super().forward([q2, k2], position_ids[:, 1, :])
        else:
            q1, k1 = super().forward([q1, k1], position_ids)
        query_states = torch.concat([q1, q2], dim=(q1.ndim - 1))
        key_states = torch.concat([k1, k2], dim=(k1.ndim - 1))
    
        return query_states, key_states


class RopeDynamicNTKScalingPositionEncoding(RopePositionEncoding):
    '''使用Dynamic NTK scaling的rope'''
    def __init__(self, 
                 embedding_size: int, 
                 max_position: int=2048, 
                 rope_rank: Literal['adjacent', 'updown', 'rotate_half']='adjacent',
                 scaling_factor: float=1.0, 
                 rope_theta: float=10000.0, 
                 **kwargs):
        self.scaling_factor_raw = scaling_factor
        scaling_factor = 1.0  # 仅在超长时候能使用的到
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    def _set_inv_freq_cache(self, seq_len, device=None):
        # 根据transformer中llama代码，dynamic时候需要seq_len > self.max_seq_len_cached才执行scaling_factor
        self.ntk_alpha = (self.scaling_factor_raw * seq_len / self.max_position) - (self.scaling_factor_raw - 1)
        return super()._set_inv_freq_cache(seq_len, device)


class RopeLlama3PositionEncoding(RopePositionEncoding):
    '''使用llama3的rope'''
    def __init__(self, 
                 embedding_size: int, 
                 max_position: int=2048, 
                 rope_rank: Literal['adjacent', 'updown', 'rotate_half']='adjacent',
                 scaling_factor: float=1.0, 
                 rope_theta: float=10000.0, 
                 **kwargs):
        self.low_freq_factor = kwargs["low_freq_factor"]  # `1` in the original implementation
        self.high_freq_factor = kwargs["high_freq_factor"]  # `4` in the original implementation
        self.old_context_len = kwargs["original_max_position_embeddings"]  # `8192` in the original implementation
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    def _set_inv_freq_cache(self, seq_len, device=None):
        base = self.rope_theta
        if (self.ntk_alpha is not None) and (self.ntk_alpha != 1):
            base = base * self.ntk_alpha ** (self.embedding_size / (self.embedding_size-2))
        
        inv_freq = torch.exp(torch.arange(0, self.embedding_size, 2).float() * (-math.log(base) / self.embedding_size))

        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / self.scaling_factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (self.high_freq_factor - self.low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / self.scaling_factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        return inv_freq_llama    


class RopeDynamicNTKScalingQwenPositionEncoding(RopePositionEncoding):
    '''使用Dynamic NTK scaling的rope (Qwen版)'''
    def _set_inv_freq_cache(self, seq_len, device=None):
        context_value = math.log(seq_len / self.max_position, 2) + 1
        ntk_alpha = max(2 ** math.ceil(context_value) - 1, 1)
        if ntk_alpha != self.ntk_alpha:
            self.ntk_alpha = ntk_alpha
        return super()._set_inv_freq_cache(seq_len, device)


class RopeYarnPositionEncoding(RopePositionEncoding):
    '''DeepSeekV2中使用'''
    def __init__(
        self,
        embedding_size,
        max_position=2048,
        rope_rank: Literal['adjacent', 'updown', 'rotate_half']='adjacent',
        scaling_factor=1.0,
        rope_theta=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
        **kwargs
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    # Inverse dim formula to find dim based on number of rotations
    @staticmethod
    def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    def yarn_find_correction_range(self, low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(self.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(self.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _set_inv_freq_cache(self, seq_len, device='cpu'):
        dim = self.embedding_size

        freq_extra = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        low, high = self.yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, self.rope_theta, self.original_max_position_embeddings)
        inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        # self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._mscale = float(self.yarn_get_mscale(self.scaling_factor, self.mscale) / self.yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_and_compute(self, x, cos, sin, position_ids, unsqueeze_dim=1):
        b, h, s, d = x.shape
        x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        x2 = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)  # rotate_half
        return x * cos + x2 * sin

    def compute_cos_sin(self, qk, position_ids):
        cos, sin =  super().compute_cos_sin(qk, position_ids)
        cos = cos * self._mscale
        sin = sin * self._mscale
        return cos, sin
    

class RopeMropePositionEncoding(RopePositionEncoding):
    '''qwen2vl中使用'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = kwargs.get('mrope_section')

    def compute_cos_sin(self, qk, position_ids):
        cos, sin =  super().compute_cos_sin(qk, position_ids)
        mrope_section = self.mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        return cos, sin

        
ROPE_ENCODGING_MAP = {
    None: RopePositionEncoding,
    'linear': RopeLinearScalingPositionEncoding,
    'dynamic': RopeDynamicNTKScalingPositionEncoding,
    'dynamic_qwen': RopeDynamicNTKScalingQwenPositionEncoding,
    'llama3': RopeLlama3PositionEncoding,
    'yarn': RopeYarnPositionEncoding,
    'mrope': RopeMropePositionEncoding,
    'glm': RopeGlmPositionEncoding
}


class XlnetPositionsEncoding(nn.Module):
    '''Xlnet, transformer_xl使用的相对位置编码
       和SinusoidalPositionEncoding区别是一个是间隔排列, 一个是前后排列
    '''
    def __init__(self, embedding_size:int):
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