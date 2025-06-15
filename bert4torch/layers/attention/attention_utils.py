import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from bert4torch.snippets import log_warn_once, is_flash_attn_available
import inspect


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
        return torch.all(torch.tril(attention_mask_4d) == attention_mask_4d)
    
    # 对1左侧的0全部补齐为1
    cummax = torch.cummax(attention_mask_4d, dim=-1)[0]
    not_all_0 = (attention_mask_4d.sum(dim=-1, keepdim=True) > 0).int()
    tril_mask_fill_left0 = (cummax < 1).int() * not_all_0 | torch.tril(attention_mask_4d.int())
    return torch.all(tril_mask_fill_left0 == attention_mask_4d.int())


def eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.FloatTensor, 
    key_states: torch.FloatTensor, 
    value_states: torch.FloatTensor, 
    attention_mask: torch.Tensor,
    return_dict_name: List[str] = None,
    **kwargs,
) -> torch.Tensor:
    '''qkv attention: torch原生实现'''
    # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

    # 相对位置编码
    attention_scores = module.apply_relative_pos_emb(query_states, key_states, attention_scores)

    if module.attention_scale:
        # 是否进行attention scale
        attention_scores = module.apply_attention_scale(attention_scores)
    
    # 执行attention mask，对于mask为0部分的attention mask，
    # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
    if attention_mask is not None:
        # attention_mask = attention_mask * attention_mask.squeeze(-2).unsqueeze(-1)  # deberta_v2中使用，但是不使用也不影响
        # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)  # 下一行的另一种写法
        attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min  # 原来逻辑是-10000，所以传入的mask的非padding部分为1, padding部分为0
        attention_scores = attention_scores + attention_mask

    # 将attention score 归一化到0-1
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_probs = module.dropout(attention_probs)
    context_layer = torch.matmul(attention_probs, value_states)  # [batch_size, num_attention_heads, query_len, attention_head_size]

    if return_dict_name:
        return {name: locals()[name] for name in return_dict_name}
    return context_layer, attention_scores
    

def sdpa_attention_forward(
    module: torch.nn.Module,
    query_states: torch.FloatTensor, 
    key_states: torch.FloatTensor, 
    value_states: torch.FloatTensor, 
    attention_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    '''sdpa: torch2.0新特性'''
    # 适用于qlen=klen, 测试下来is_causal=True训练更快
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    is_causal = False
    if (key_states.shape[2] == query_states.shape[2]) and \
        (torch.all(attention_mask == 1) or is_causal_mask(attention_mask, ignore_left_padding=False)):
        is_causal = True
    min_dtype = torch.finfo(query_states.dtype).min
    attention_mask = (1.0 - attention_mask) * min_dtype
    attention_mask = attention_mask.mul(~torch.all(attention_mask == min_dtype, dim=-1, keepdim=True))  # 将padding部分的mask值变为0
    context_layer = F.scaled_dot_product_attention(
        query_states, 
        key_states, 
        value_states, 
        attn_mask = None if is_causal else attention_mask,
        dropout_p = module.attention_probs_dropout_prob if module.training else 0.0,
        scale = module.scaling,
        is_causal = is_causal
        )
    return context_layer, None


def flash_attention_forward(
    module: torch.nn.Module,
    query_states: torch.FloatTensor, 
    key_states: torch.FloatTensor, 
    value_states: torch.FloatTensor, 
    attention_mask: torch.Tensor, 
    past_key_value: Union[Tuple[torch.Tensor]], 
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

        kv_seq_len = key_states.shape[1]  # [btz, n_heads, seq_len, d_head]
        use_sliding_windows = (_flash_supports_window_size and module.sliding_window is not None and kv_seq_len > module.sliding_window)

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
        slicing_tokens = -module.sliding_window

        past_key = past_key_value[0][:, :, slicing_tokens:, :].contiguous()
        past_value = past_key_value[1][:, :, slicing_tokens:, :].contiguous()
        past_key_value = (past_key, past_value)

        if past_key.shape[-2] != module.sliding_window:
            raise ValueError(
                f"past key must have a shape of (`batch_size, num_heads, module.sliding_window-1, head_dim`), got"
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
    
    dropout = 0.0 if not module.training else module.attention_probs_dropout_prob
    
    is_causal = is_causal_mask(attention_mask)
    query_length = query_states.shape[-2]  # [batch_size, num_attention_heads, query_len, attention_head_size]
    if (not is_causal) and (attention_mask.shape[1:3] == torch.Size([1,1])):
        query_states, key_states, value_states = _transpose(query_states, key_states, value_states)
        use_sliding_windows = _use_sliding_windows()
        if use_sliding_windows:
            key_states, value_states, past_key_value, attention_mask = _run_sliding_windows(key_states, value_states, past_key_value, attention_mask)

        # flash attention目前仅支持key_padding_mask
        attn_mask = attention_mask[:,0,0,:]  # 将4维的attention_mask降低为2维
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            module, query_states, key_states, value_states, attn_mask, query_length)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad = flash_attn_varlen_func(
            query_states, 
            key_states, 
            value_states, 
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_k, 
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k, 
            dropout_p=dropout, 
            softmax_scale=module.scaling, 
            causal=False, 
            window_size=(module.sliding_window, module.sliding_window) if use_sliding_windows else (-1, -1)
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    elif is_causal:
        query_states, key_states, value_states = _transpose(query_states, key_states, value_states)
        # attention_mask满足下三角的causal
        use_sliding_windows = _use_sliding_windows()
        if use_sliding_windows:
            key_states, value_states, past_key_value, attention_mask = _run_sliding_windows(key_states, value_states, past_key_value, attention_mask)

        attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=module.scaling, causal=True,
                                        window_size=(module.sliding_window, module.sliding_window) if use_sliding_windows else (-1, -1))
    
    elif is_causal:
        # 使用torch的attention计算
        log_warn_once( 'Flash Attention only support key_padding_mask, use eager_attention_forward instead.')
        module._attn_implementation = 'eager'
        return attn_output, None

    return attn_output.transpose(1,2), None