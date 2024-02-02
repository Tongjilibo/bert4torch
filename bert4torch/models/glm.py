from bert4torch.models.transformer import Decoder
from bert4torch.layers import LayerNorm, BertLayer, BlockIdentity
from bert4torch.snippets import delete_arguments
from bert4torch.activations import get_activation
import torch
from torch import nn
import copy


class GLM(Decoder):
    '''GLM: https://github.com/THUDM/GLM, ChatGLM-6B: https://github.com/THUDM/ChatGLM-6B
    Unilm设计, 可定义为GLM(UniLM_MASK, BERT)但是要求传入segement_ids比较麻烦, 这里继承LM_MASK并使用get_masks()重新构造attention_mask
    模型结构特点：
    1) rotary使用的updown+position_encoding_2d
    2) qkv合并成一个权重convert时不是concat在一起的
    3) attention_mask类似于Unilm, 最后一个token仅能访问之前的, 之前的tokens可以互相访问
    4) 跳跃连接有权重设计
    5) embedding之后没有layernorm
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': True, 'is_decoder': True, 'final_layernorm': True})
        super().__init__(*args, **kwargs)
        self.bos_token_id, self.mask_token_id, self.gmask_token_id = kwargs.get('bos_token_id'), kwargs.get('mask_token_id'), kwargs.get('gmask_token_id')
        self.position_encoding_2d = kwargs.get('position_encoding_2d', True)
        del self.embeddings.layerNorm
        layer = self.GlmBlock(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                            'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'num_hidden_layers', **kwargs))
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = torch.nn.LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12))
        self.model_type = 'glm'

    def load_trans_ckpt(self, checkpoint, prefix='transformer'):
        state_dict = super().load_trans_ckpt(checkpoint)
        # weight bias
        for i in range(self.num_hidden_layers):
            mapping = {
                f'{prefix}.layers.{i}.attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'{prefix}.layers.{i}.attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias',
                # int8和int4的weight_scale权重
                f'{prefix}.layers.{i}.attention.query_key_value.weight_scale': 'decoderLayer.{}.multiHeadAttention.{}.weight_scale'  
            }
            for old_key, new_key in mapping.items():
                if (qkv := state_dict.get(old_key)) is None:
                    continue
                qkv = torch.split(qkv, self.attention_head_size, 0)
                q, k, v = qkv[0::3], qkv[1::3], qkv[2::3]
                q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[new_key.format(i, i_k)] = i_v
                state_dict.pop(old_key)
        return state_dict
    
    def save_trans_ckpt(self, prefix='transformer'):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'{prefix}.layers.{i}.attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'{prefix}.layers.{i}.attention.query_key_value.bias',
                # int8和int4的weight_scale权重
                'decoderLayer.{}.multiHeadAttention.{}.weight_scale': f'{prefix}.layers.{i}.attention.query_key_value.weight_scale'
            }
            for old_key, new_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if old_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(old_key.format(i, i_k)).split(self.attention_head_size, 0))
                if qkv:
                    state_dict[new_key] = torch.cat([torch.cat(i) for i in zip(*qkv)])
        return state_dict
    
    def variable_mapping(self, prefix='transformer'):
        # 映射到权重格式
        mapping = {
            'LayerNormFinal.weight': f"{prefix}.final_layernorm.weight",
            'LayerNormFinal.bias': f"{prefix}.final_layernorm.bias",
            'lm_head.weight': "lm_head.weight",
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
        if self.position_encoding_2d:
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
            if self.position_encoding_2d:  # [btz, 2, 1]
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
       
    class GlmBlock(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_hidden_layers = kwargs['num_hidden_layers']
            hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-5)
            self.attnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)
            self.ffnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)

        def forward(self, hidden_states=None, attention_mask=None, past_key_value=None, **model_kwargs):
            # 和bert区别有两点, 一个是有alpha, 还有一个是跳跃链接用的是经过了layernorm后的
            x = self.attnLayerNorm(hidden_states)
            alpha = (2 * self.num_hidden_layers) ** 0.5
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
            hidden_states = x * alpha + self_attn_output[0]

            x = self.ffnLayerNorm(hidden_states)
            hidden_states = x *alpha +  self.feedForward(x)

            if self.is_decoder and model_kwargs.get('use_states', False):
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs

        
class GLM2(GLM):
    """CHATGLM2-6B: https://github.com/THUDM/ChatGLM2-6B
    主要修改：1) 不使用Unilm式的mask
             2) flash_attention
             3) multi_query_attention
    """
    def __init__(self, *args, **kwargs):
        kwargs.update({'norm_mode': 'rmsnorm', 'pre_layernorm': True})
        super().__init__(*args, **kwargs)
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-5), norm_mode='rmsnorm', bias=False)
        self.model_type = 'glm2'

    def load_trans_ckpt(self, checkpoint, prefix='transformer.encoder'):
        state_dict = super().load_trans_ckpt(checkpoint, prefix)
        # weight bias
        for i in range(self.num_hidden_layers):
            mapping = {
                f'{prefix}.layers.{i}.self_attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'{prefix}.layers.{i}.self_attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias',
                f'{prefix}.layers.{i}.self_attention.query_key_value.weight_scale': 'decoderLayer.{}.multiHeadAttention.{}.weight_scale'
            }
            for old_key, new_key in mapping.items():
                if (qkv := state_dict.get(old_key)) is None:
                    continue
                inner_dim = (qkv.shape[0]-self.hidden_size) // 2
                q, k, v = torch.split(qkv, [self.hidden_size, inner_dim, inner_dim], 0)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[new_key.format(i, i_k)] = i_v
                state_dict.pop(old_key)
        return state_dict

    def save_trans_ckpt(self, prefix='transformer.encoder'):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            mapping = {
                'decoderLayer.{}.multiHeadAttention.{}.weight': f'{prefix}.layers.{i}.self_attention.query_key_value.weight',
                'decoderLayer.{}.multiHeadAttention.{}.bias': f'{prefix}.layers.{i}.self_attention.query_key_value.bias',
                'decoderLayer.{}.multiHeadAttention.{}.weight_scale': f'{prefix}.layers.{i}.self_attention.query_key_value.weight_scale'
            }
            for old_key, new_key in mapping.items():
                qkv = []
                for i_k in ['q', 'k', 'v']:
                    if old_key.format(i, i_k) in state_dict:
                        qkv.append(state_dict.pop(old_key.format(i, i_k)))
                if qkv:
                    state_dict[new_key] = torch.cat(qkv)
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
    
    class GlmBlock(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-5)
            self.attnLayerNorm = LayerNorm(hidden_size, eps=eps, norm_mode='rmsnorm', bias=False)
            self.ffnLayerNorm = LayerNorm(hidden_size, eps=eps, norm_mode='rmsnorm', bias=False)
            self.multiHeadAttention.o.register_parameter('bias', None)
            self.feedForward.intermediateDense.register_parameter('bias', None)
            self.feedForward.outputDense.register_parameter('bias', None)
