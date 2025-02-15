from bert4torch.models.bert import BERT
import torch
from bert4torch.snippets import delete_arguments
from bert4torch.layers import LayerNorm, BlockIdentity
try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
except:
    pass


class ModernBert(BERT):
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'mlp_type': 'LlamaFeedForward', 'bias':False, 'p_bias': 'rotary',
                       'attn_type': 'ModernBertAttention', 'norm_mode': 'torch_buildin', 
                       'pre_layernorm':True})
        super(ModernBert, self).__init__(*args, **kwargs)
        self.model_type = 'modernbert'
        self.local_attention = kwargs['local_attention']
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12), 
                                        conditional_size=self.conditional_size, 
                                        norm_mode=kwargs.get('norm_mode', 'torch_buildin'),
                                        weight=kwargs.get('weight', True), 
                                        bias=kwargs.get('bias', True))
        self.encoderLayer[0].attnLayerNorm = BlockIdentity(return_args_index=set([0]))
        self.mlmDense.register_parameter('bias', None)
        self.mlmLayerNorm.register_parameter('bias', None)

    def _update_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)

        return sliding_window_mask

    def apply_embeddings(self, *inputs, **model_kwargs):
        model_kwargs['sliding_window_mask'] = self._update_attention_mask(model_kwargs['attention_mask'])
        model_kwargs = super().apply_embeddings(*inputs, **model_kwargs)
        return model_kwargs
    
    def apply_final_layers(self, **model_kwargs):
        last_hidden_state = model_kwargs['encoded_layers'][-1]
        model_kwargs['encoded_layers'][-1] = self.LayerNormFinal(last_hidden_state)
        return super().apply_final_layers(**model_kwargs)
    
    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        # weight bias
        for i in range(self.num_hidden_layers):
            ckpt_key, model_key = f'model.layers.{i}.attn.Wqkv.weight', 'encoderLayer.{}.multiHeadAttention.{}.weight'
            # 如果当前ckpt不存在该key，则跳过
            if (qkv := state_dict.get(ckpt_key)) is None:
                continue
            qkv = torch.chunk(qkv, 3, dim=0)
            for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                state_dict[model_key.format(i, i_k)] = i_v
            state_dict.pop(ckpt_key)
            
            mlp_Wi = state_dict.pop(f'model.layers.{i}.mlp.Wi.weight')
            intermediateDense, intermediateDense2 = torch.chunk(mlp_Wi, 2, dim=0)
            state_dict[f'encoderLayer.{i}.feedForward.intermediateDense.weight'] = intermediateDense
            state_dict[f'encoderLayer.{i}.feedForward.intermediateDense2.weight'] = intermediateDense2
        return state_dict
    
    def save_trans_ckpt(self):
        '''把q,k,v合并成qkv, 以便于transformers包加载'''
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            model_key, ckpt_key = 'encoderLayer.{}.multiHeadAttention.{}.weight', f'model.layers.{i}.attn.Wqkv.weight'
            qkv = []
            for i_k in ['q', 'k', 'v']:
                qkv.append(state_dict.pop(model_key.format(i, i_k)))
            if qkv:
                state_dict[ckpt_key] = torch.cat(qkv)
            
            intermediateDense = state_dict[f'encoderLayer.{i}.feedForward.intermediateDense.weight']
            intermediateDense2 = state_dict[f'encoderLayer.{i}.feedForward.intermediateDense2.weight']
            state_dict[f'model.layers.{i}.mlp.Wi.weight'] = torch.cat([intermediateDense, intermediateDense2])
        return state_dict

    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'model.embeddings.tok_embeddings.weight',
            'embeddings.layerNorm.weight': 'model.embeddings.norm.weight',
            'LayerNormFinal.weight': f'model.final_norm.weight',
            'mlmDense.weight': 'head.dense.weight',
            'mlmLayerNorm.weight': 'head.norm.weight',
            'mlmBias': 'decoder.bias'
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'model.layers.{i}.'
            mapping.update({
                f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attn.Wo.weight',
                f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'mlp.Wo.weight',
                f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'mlp_norm.weight',
                })
            if i > 0:
                mapping[f'encoderLayer.{i}.attnLayerNorm.weight'] = prefix_i + 'attn_norm.weight'

        return mapping

    def load_variable(self, variable, ckpt_key, model_key):
        # 加载单个变量的函数
        if ckpt_key in {
            'model.embeddings.tok_embeddings.weight',
            'predictions.bias',
            'predictions.decoder.weight',
            'predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        else:
            return variable