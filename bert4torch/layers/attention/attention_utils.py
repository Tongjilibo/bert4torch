import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Type, Callable
from bert4torch.snippets import log_warn_once, is_flash_attn_available, is_xformers_available, create_registrar
from bert4torch.models.modeling_utils import get_proper_attn_implementation
import inspect


if is_xformers_available():
    from xformers import ops as xops


class AttentionFunctionDict(Dict[str, Type[Callable]]):
    def __getitem__(self, key):
        key = get_proper_attn_implementation(key)
        return super().__getitem__(key)


ALL_ATTENTION_FUNCTIONS: AttentionFunctionDict = AttentionFunctionDict()
regiister_attn_forward = create_registrar(ALL_ATTENTION_FUNCTIONS)


if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def is_causal_mask(attention_mask_4d:torch.Tensor, ignore_left_padding=True):
    '''判断一个矩阵是不是下三角阵
    :param attention_mask_4d: torch.Tensor, 4维的attention mask, [btz, n_heads, seq_q, seq_k]
    :param ignore_left_padding: bool, 是否忽略left_padding部分的mask来比较, True表示忽略
    '''
    if ignore_left_padding:
        # left padding的下三角mask认为是True
        return torch.all(torch.tril(attention_mask_4d) == attention_mask_4d).item()
    
    # 对1左侧的0全部补齐为1
    cummax = torch.cummax(attention_mask_4d, dim=-1)[0]
    not_all_0 = (attention_mask_4d.sum(dim=-1, keepdim=True) > 0).int()
    tril_mask_fill_left0 = (cummax < 1).int() * not_all_0 | torch.tril(attention_mask_4d.int())
    return torch.all(tril_mask_fill_left0 == attention_mask_4d.int()).item()


def is_01_mask(attention_mask:torch.Tensor) -> bool:
    '''是否是01的mask, bert4torch的, transformers是已经使用min填充后的'''
    return ((attention_mask == 0) | (attention_mask == 1)).all().item()


def is_1_mask(attention_mask:torch.Tensor) -> bool:
    return torch.all(attention_mask == 1).item()


def repeat_kv(hidden_states:torch.Tensor, n_rep:int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@regiister_attn_forward(name='eager')
def eager_attention_forward(
    module: torch.nn.Module,
    query: torch.FloatTensor, 
    key: torch.FloatTensor, 
    value: torch.FloatTensor, 
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    return_dict_name: List[str] = None,
    **kwargs,
) -> torch.Tensor:
    '''qkv attention: torch原生实现'''
    # multi_query_attention
    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    # 相对位置编码
    attention_scores = module.apply_relative_pos_emb(query, key, attention_scores)

    # scaling, 这里融合到apply_relative_pos_emb中去
    # attention_scores = attention_scores * scaling
    
    # 执行attention mask，对于mask为0部分的attention mask，
    # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
    if attention_mask is not None:
        if is_01_mask(attention_mask):
            # attention_mask = attention_mask * attention_mask.squeeze(-2).unsqueeze(-1)  # deberta_v2中使用，但是不使用也不影响
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)  # 下一行的另一种写法
            attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min  # 原来逻辑是-10000，所以传入的mask的非padding部分为1, padding部分为0
        attention_scores = attention_scores + attention_mask

    # 将attention score 归一化到0-1
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_probs = module.dropout(attention_probs)
    attn_output = torch.matmul(attention_probs, value)  # [batch_size, num_attention_heads, query_len, attention_head_size]

    attn_output = attn_output.transpose(1, 2).contiguous()
    if return_dict_name:
        return {name: locals()[name] for name in return_dict_name}
    return attn_output, attention_scores
    

@regiister_attn_forward(name='sdpa')
def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.FloatTensor, 
    key: torch.FloatTensor, 
    value: torch.FloatTensor, 
    attention_mask: torch.Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:
    '''sdpa: torch2.0新特性'''
    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if attention_mask is not None and is_01_mask(attention_mask):
        # 1. bert4torch风格的attention_mask：0为padding，1为非padding
        if is_1_mask(attention_mask):
            # 1.1 bert的全为1的attention_mask: [[[[1,1,1,1,1,1]]]]
            is_causal = False
        elif is_causal is None:
            # 1.2 为01格式的4d_attention_mask, 需要判断是否满足causal的下三角格式
            # 1.2.1 extend_with_language_model会修改mask为下三角
            # 1.2.2 extend_with_unified_language_model会修改mask为UniLM的左侧为1，右边是下三角的形式
            # 对于CausalModel，当且仅当step=1，即prefill阶段是is_causal=True
            is_causal = (query.shape[2] == key.shape[2]) and is_causal_mask(attention_mask, ignore_left_padding=False)

        min_dtype = torch.finfo(query.dtype).min
        attention_mask = (1.0 - attention_mask) * min_dtype
        attention_mask = attention_mask.mul(~torch.all(attention_mask == min_dtype, dim=-1, keepdim=True))  # 将padding部分的mask值变为0
    
    else:
        # 2. transformer风格
        # 2.1 attention_mask不为None, 且是transformer格式的4d_attention_mask, 0和-inf组成
        # 2.2 attention_mask为None时，is_causal=True
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = F.scaled_dot_product_attention(
        query, 
        key, 
        value, 
        attn_mask = None if is_causal else attention_mask,
        dropout_p = dropout,
        scale = scaling,
        is_causal = is_causal  # is_causal速度更块
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


@regiister_attn_forward(name='flash_attention_4')
@regiister_attn_forward(name='flash_attention_3')
@regiister_attn_forward(name='flash_attention_2')
@regiister_attn_forward(name='flash_attention')
def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.FloatTensor, 
    key: torch.FloatTensor, 
    value: torch.FloatTensor, 
    attention_mask: torch.Tensor, 
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    past_key_value: Union[Tuple[torch.Tensor]]=None, 
    **kwargs,
) -> torch.Tensor:
    """ flash_attn，参考transformers中的调用
    """
    def _get_unpad_data(attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    def _upad_input(self, query_states, key_states, value_states, attention_mask, query_length):       
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_states.shape

        key_states = index_first_axis(key_states.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_states = index_first_axis(value_states.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_states = index_first_axis(query_states.reshape(batch_size * kv_seq_len, module.num_attention_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_states.device)  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_states = query_states.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask)

        return (query_states, key_states, value_states, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k),)
    
    def _use_sliding_windows():
        if (module.max_window_layers is not None) and (module.layer_idx >= module.max_window_layers):
            return False

        kv_seq_len = key.shape[1]  # [btz, n_heads, seq_len, d_head]
        use_sliding_windows = (_flash_supports_window_size and sliding_window is not None and kv_seq_len > sliding_window)

        if use_sliding_windows and not _flash_supports_window_size:
            log_warn_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )
            use_sliding_windows = False

        if use_sliding_windows and past_key_value is not None and past_key_value[0].shape[2] > 0:
            use_sliding_windows = True
        return use_sliding_windows

    def _run_sliding_windows(key_states, value_states, past_key_value, attention_mask):
        '''sliding_window部分'''
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        slicing_tokens = -sliding_window

        past_key = past_key_value[0][:, :, slicing_tokens:, :].contiguous()
        past_value = past_key_value[1][:, :, slicing_tokens:, :].contiguous()
        past_key_value = (past_key, past_value)

        if past_key.shape[-2] != sliding_window:
            raise ValueError(
                f"past key must have a shape of (`batch_size, num_heads, sliding_window-1, head_dim`), got"
                f" {past_key.shape}"
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, slicing_tokens:, slicing_tokens:]
        
        key_states = key_states[:, slicing_tokens:, :, :].contiguous()
        value_states = value_states[:, slicing_tokens:, :, :].contiguous()
        return key_states, value_states, past_key_value, attention_mask

    def _transpose(query_states, key_states, value_states):
        # [batch_size, query_len, num_attention_heads, attention_head_size]
        query_states = query_states.transpose(1,2)
        key_states = key_states.transpose(1,2)
        value_states = value_states.transpose(1,2)
        return query_states, key_states, value_states
        
    is_causal = is_causal_mask(attention_mask)
    query_length = query.shape[-2]  # [batch_size, num_attention_heads, query_len, attention_head_size]
    if (not is_causal) and (attention_mask.shape[1:3] == torch.Size([1,1])):
        query, key, value = _transpose(query, key, value)
        use_sliding_windows = _use_sliding_windows()
        if use_sliding_windows:
            key, value, past_key_value, attention_mask = _run_sliding_windows(key, value, past_key_value, attention_mask)

        # flash attention目前仅支持key_padding_mask
        attn_mask = attention_mask[:,0,0,:]  # 将4维的attention_mask降低为2维
        batch_size = query.shape[0]
        query, key, value, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            module, query, key, value, attn_mask, query_length)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad = flash_attn_varlen_func(
            query, 
            key, 
            value, 
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_k, 
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k, 
            dropout_p=dropout, 
            softmax_scale=scaling, 
            causal=False, 
            window_size=(sliding_window, sliding_window) if use_sliding_windows else (-1, -1)
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    elif is_causal:
        query, key, value = _transpose(query, key, value)
        # attention_mask满足下三角的causal
        use_sliding_windows = _use_sliding_windows()
        if use_sliding_windows:
            key, value, past_key_value, attention_mask = _run_sliding_windows(key, value, past_key_value, attention_mask)

        attn_output = flash_attn_func(query, key, value, dropout, softmax_scale=scaling, causal=True,
                                        window_size=(sliding_window, sliding_window) if use_sliding_windows else (-1, -1))
    
    elif is_causal:
        # 使用torch的attention计算
        log_warn_once( 'Flash Attention only support key_padding_mask, use eager_attention_forward instead.')
        module._attn_implementation = 'eager'
        return attn_output, None

    return attn_output, None


@regiister_attn_forward
def xformers_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_output = xops.memory_efficient_attention(query, key, value, attn_bias=xops.LowerTriangularMask())
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
