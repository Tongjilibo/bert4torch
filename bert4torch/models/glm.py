from bert4torch.models.transformer import Decoder
from bert4torch.layers import LayerNorm
from bert4torch.snippets import delete_arguments
import torch


class GLM(Decoder):
    '''GLM: https://github.com/THUDM/GLM, ChatGLM-6B: https://github.com/THUDM/ChatGLM-6B
    Unilm设计, 可定义为GLM(UniLM_MASK, BERT)但是要求传入segement_ids比较麻烦, 这里继承LM_MASK并使用get_masks()重新构造attention_mask
    模型结构特点：
    1) rotary使用的updown
    2) qkv合并成一个权重convert时不是concat在一起的
    3) attention_mask类似于Unilm, 最后一个token仅能访问之前的, 之前的tokens可以互相访问
    4) 跳跃连接有权重设计
    5) embedding之后没有layernorm
    '''
    _no_split_modules = ["GlmLayer"]
    def __init__(self, *args, layer_type='GlmLayer', **kwargs):
        kwargs.update({'layer_type': layer_type, 'p_bias': 'rotary', 'weight': True, 'is_decoder': True, 'final_layernorm': True})
        super().__init__(*args, **kwargs)
        self.bos_token_id, self.mask_token_id, self.gmask_token_id = kwargs.get('bos_token_id'), kwargs.get('mask_token_id'), kwargs.get('gmask_token_id')
        self.rope_scaling_type = kwargs.get('rope_scaling', {}).get('type')
        del self.embeddings.layerNorm
        self.LayerNormFinal = torch.nn.LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12))
        self.model_type = 'glm'

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        # weight bias
        for i in range(self.num_hidden_layers):
            mapping = {
                f'transformer.layers.{i}.attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.layers.{i}.attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias',
                # int8和int4的weight_scale权重
                f'transformer.layers.{i}.attention.query_key_value.weight_scale': 'decoderLayer.{}.multiHeadAttention.{}.weight_scale'  
            }
            for ckpt_key, model_key in mapping.items():
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                qkv = torch.split(qkv, self.attention_head_size, 0)
                q, k, v = qkv[0::3], qkv[1::3], qkv[2::3]
                q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.layers.{i}.attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.layers.{i}.attention.query_key_value.bias',
                # int8和int4的weight_scale权重
                'decoderLayer.{}.multiHeadAttention.{}.weight_scale': f'transformer.layers.{i}.attention.query_key_value.weight_scale'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if model_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(model_key.format(i, i_k)).split(self.attention_head_size, 0))
                if qkv:
                    state_dict[ckpt_key] = torch.cat([torch.cat(i) for i in zip(*qkv)])
        return state_dict
    
    def variable_mapping(self, prefix='transformer'):
        # 映射到权重格式
        mapping = {
            'LayerNormFinal.weight': f"{prefix}.final_layernorm.weight",
            'LayerNormFinal.bias': f"{prefix}.final_layernorm.bias",
            'lm_head.weight': 'lm_head.weight' if self.with_lm and not self.tie_word_embeddings else 'model.embed_tokens.weight',
            'embeddings.word_embeddings.weight': 'transformer.word_embeddings.weight'}

        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layers.%d.' % i
            mapping.update({
                f'decoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'input_layernorm.weight',
                f'decoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'input_layernorm.bias',
                f'decoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'post_attention_layernorm.weight',
                f'decoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'post_attention_layernorm.bias',
                f'decoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight',
                f'decoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias',
                f'decoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'mlp.dense_h_to_4h.weight',
                f'decoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'mlp.dense_h_to_4h.bias',
                f'decoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'mlp.dense_4h_to_h.weight',
                f'decoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'mlp.dense_4h_to_h.bias',
                # 加载int4和int8使用
                f'decoderLayer.{i}.multiHeadAttention.o.weight_scale': prefix_i + 'attention.dense.weight_scale',
                f'decoderLayer.{i}.feedForward.intermediateDense.weight_scale': prefix_i + 'mlp.dense_h_to_4h.weight_scale',
                f'decoderLayer.{i}.feedForward.outputDense.weight_scale': prefix_i + 'mlp.dense_4h_to_h.weight_scale',
                })
        return mapping

    def get_masks(self, attention_mask, context_lens, prepad_lens):
        '''调整mask使得在content_lens前是bi_attention'''
        for i, (prepad_len, context_len) in enumerate(zip(prepad_lens, context_lens)):
            attention_mask[i, :, :, prepad_len:context_len] = 1
        return attention_mask
        
    def get_position_ids(self, position_ids, seq_len, context_lens, mask_positions, prepad_lens, gmask=False):
        '''不使用cache时候的postion_ids'''
        if position_ids.shape[0] == 1:
            position_ids = position_ids.repeat(len(context_lens), 1)
        if self.rope_scaling_type == 'glm':
            # 初始版本中这里也有not gmask
            for i, context_length in enumerate(context_lens):
                position_ids[i, context_length:] = mask_positions[i] - prepad_lens[i]
            block_position_ids = [torch.cat((torch.zeros(context_len, dtype=torch.long).to(position_ids),
                                            torch.arange(seq_len-context_len, dtype=torch.long).to(position_ids) + 1)) for context_len in context_lens]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            if not gmask:
                for i, context_length in enumerate(context_lens):
                    position_ids[context_length:] = mask_positions[i] - prepad_lens[i]
        return position_ids

    def prepare_inputs(self, *inputs, **model_kwargs):
        '''对attention_mask(参考unilm方式)和position_ids做处理'''
        token_ids = model_kwargs['past_token_ids'] if model_kwargs.get('past_token_ids') is not None else inputs[0]
        mask_token = self.mask_token_id if self.mask_token_id in token_ids else self.gmask_token_id  # 倒数第2位
        use_gmask = False if self.mask_token_id in token_ids else True
        position_ids = model_kwargs['position_ids']
        device = position_ids.device
        seqs = token_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]
        context_lens = [seq.index(self.bos_token_id) for seq in seqs]  # bos_token_id是倒数第一位
        seq_len = token_ids.shape[1]

        # 1) generation阶段use_states=True且step>0的时候(用cache)
        # 这里用inputs[0].shape[1] == 1来判断是不是last_token, chatglm过tokenize出来最后会以[mask_token_id, bos_token_id]结尾, 长度>1
        if model_kwargs.get('use_states', False) and (inputs[0].shape[1] == 1) and (model_kwargs.get('past_key_values') is not None):
            if self.rope_scaling_type == 'glm':  # [btz, 2, 1]
                position_ids = torch.tensor([[mask_position, seq_len - context_len] for mask_position, context_len in
                                            zip(mask_positions, context_lens)], dtype=torch.long, device=device).unsqueeze(-1)
            else:  # [btz, 1]
                position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long, device=device).unsqueeze(-1)
            model_kwargs['position_ids'] = position_ids
        # 1) train阶段；2) generation阶段use_states=False；3) use_states=True且step=0的时候
        else:
            prepad_lens = [(ts[:l]==self.pad_token_id).sum().item() for l, ts in zip(context_lens, token_ids)]
            model_kwargs['attention_mask'] = self.get_masks(model_kwargs['attention_mask'], context_lens, prepad_lens)
            model_kwargs['position_ids'] = self.get_position_ids(position_ids, seq_len, context_lens, mask_positions, prepad_lens, gmask=use_gmask)
        return model_kwargs

    def apply_embeddings(self, *inputs, **model_kwargs):
        model_kwargs = super().apply_embeddings(*inputs, **model_kwargs)
        model_kwargs = self.prepare_inputs(*inputs, **model_kwargs)
        return model_kwargs
    
    def prepare_inputs_for_generation(self, *inputs, **states):
        '''为下次generate做准备'''
        output_ids = states.pop('output_ids')
        input_seqlen = states.pop('input_seqlen')
        token_ids = inputs[0]

        if output_ids.numel() == 0:
            past_token_ids = token_ids
        elif len(token_ids) == 1:  # TODO 并非使用batch
            past_token_ids = torch.cat([token_ids, output_ids], 1)
        else:
            inputs = []
            for seq_l, token_ids_i, output_ids_i in zip(input_seqlen, token_ids, output_ids):
                inputs.append(torch.cat([token_ids_i[:seq_l], output_ids_i, token_ids_i[seq_l:]]))
            past_token_ids = torch.stack(inputs)
        
        # past_token_ids: inputs+output_ids
        states['past_token_ids'] = past_token_ids
        return states

    @property
    def _layer_args(self):
        return super()._layer_args + ['num_hidden_layers']


class GLM2(GLM):
    """CHATGLM2-6B: https://github.com/THUDM/ChatGLM2-6B
    主要修改：1) 不使用Unilm式的mask
             2) flash_attention
             3) multi_query_attention
    """
    _no_split_modules = ["Glm2Layer"]
    def __init__(self, *args, **kwargs):
        kwargs.update({'layer_type': "Glm2Layer", 'norm_mode': 'rmsnorm', 'rmsnorm_fp32': 'glm', 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-5), norm_mode='rmsnorm', bias=False)
        self.model_type = 'glm2'

    def load_trans_ckpt(self, checkpoint, prefix=''):
        state_dict = super().load_trans_ckpt(checkpoint)
        # weight bias
        for i in range(self.num_hidden_layers):
            mapping = {
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight': prefix + 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias': prefix + 'decoderLayer.{}.multiHeadAttention.{}.bias',
                f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight_scale': prefix + 'decoderLayer.{}.multiHeadAttention.{}.weight_scale'
            }
            for ckpt_key, model_key in mapping.items():
                if (qkv := state_dict.get(ckpt_key)) is None:
                    continue
                inner_dim = (qkv.shape[0]-self.hidden_size) // 2
                q, k, v = torch.split(qkv, [self.hidden_size, inner_dim, inner_dim], 0)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[model_key.format(i, i_k)] = i_v
                state_dict.pop(ckpt_key)
        return state_dict

    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias',
                'decoderLayer.{}.multiHeadAttention.{}.weight_scale': f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight_scale'
            }
            for model_key, ckpt_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if model_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(model_key.format(i, i_k)))
                if qkv:
                    state_dict[ckpt_key] = torch.cat(qkv)
        return state_dict

    def variable_mapping(self, prefix='transformer.encoder'):
        mapping = super().variable_mapping(prefix)
        mapping.update({
            'embeddings.word_embeddings.weight': 'transformer.embedding.word_embeddings.weight',
            'lm_head.weight': "transformer.output_layer.weight"
        })
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layers.%d.' % i
            mapping.update({
                f'decoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'self_attention.dense.weight',
                f'decoderLayer.{i}.multiHeadAttention.o.weight_scale': prefix_i + "self_attention.dense.weight_scale",
                f'decoderLayer.{i}.feedForward.intermediateDense.weight_scale': prefix_i + "mlp.dense_h_to_4h.weight_scale",
                f'decoderLayer.{i}.feedForward.outputDense.weight_scale': prefix_i + "mlp.dense_4h_to_h.weight_scale",
                f'decoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + "self_attention.dense.weight",
            })
        return mapping

    def prepare_inputs(self, *inputs, **model_kwargs):
        return model_kwargs
