from bert4torch.models.transformer import Decoder
from bert4torch.snippets import delete_arguments
from bert4torch.layers import BertLayer, BlockIdentity
from torch import nn
import copy
import torch


class GPT(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 有segment, embedding是token、position、segment三者embedding之和
    2. embedding没有加LayerNormalization层
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(GPT, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        self.prefix = 'gpt'

    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        # CDial-GPT的[CLS]是0、[PAD]是1，不符合一般习惯，所以交换一下
        old_key = 'transformer.tokens_embed.weight'
        w = state_dict[old_key]
        state_dict[f'embeddings.word_embeddings.weight'] = torch.cat([w[1:2], w[:1], w[2:]], axis=0)
        state_dict.pop(old_key)

        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'transformer.h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for old_key, new_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(old_key)) is not None:
                    is_weight = old_key.endswith('weight')
                    qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                    for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                        state_dict[new_key.format(i, i_k)] = i_v.T if is_weight else i_v
                    state_dict.pop(old_key)

            # hdsz-hdsz的全连接
            old_key = f'transformer.h.{i}.attn.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.multiHeadAttention.o.weight'] = w.T
                state_dict.pop(old_key)

            # feed forward 第一层
            old_key = f'transformer.h.{i}.mlp.c_fc.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.intermediateDense.weight'] = w.T
                state_dict.pop(old_key)
                
            # feed forward 第二层
            old_key = f'transformer.h.{i}.mlp.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.outputDense.weight'] = w.T
                state_dict.pop(old_key)

        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.position_embeddings.weight': 'transformer.positions_embed.weight',
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'transformer.h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'transformer.h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'transformer.h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'transformer.h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'transformer.h.{i}.ln_2.bias'
            })
        return mapping


class GPT2(Decoder):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    1. 没有segment输入
    2. embedding之后没有LayerNormalization层
    3. 使用pre_layernorm, bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
    4. 有final_layernorm
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        kwargs['pre_layernorm'] = kwargs.get('pre_layernorm', True)
        kwargs['final_layernorm'] = kwargs.get('final_layernorm', True)
        super(GPT2, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        self.prefix = 'gpt2'

    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'transformer.h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'transformer.h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for old_key, new_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(old_key)) is not None:
                    is_weight = old_key.endswith('weight')
                    qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                    for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                        state_dict[new_key.format(i, i_k)] = i_v.T if is_weight else i_v
                    state_dict.pop(old_key)

            # hdsz-hdsz的全连接
            old_key = f'transformer.h.{i}.attn.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.multiHeadAttention.o.weight'] = w.T
                state_dict.pop(old_key)

            # feed forward 第一层
            old_key = f'transformer.h.{i}.mlp.c_fc.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.intermediateDense.weight'] = w.T
                state_dict.pop(old_key)
                
            # feed forward 第二层
            old_key = f'transformer.h.{i}.mlp.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.outputDense.weight'] = w.T
                state_dict.pop(old_key)

        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'transformer.wte.weight',
            'embeddings.position_embeddings.weight': 'transformer.wpe.weight',
            'LayerNormFinal.weight': 'transformer.ln_f.weight',
            'LayerNormFinal.bias': 'transformer.ln_f.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'transformer.h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'transformer.h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'transformer.h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'transformer.h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'transformer.h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'transformer.h.{i}.ln_2.bias'
            })
        return mapping


class GPT2_ML(Decoder):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    1. embedding后也有layernorm
    2. 第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super().__init__(*args, **kwargs)
        layer = self.Gpt2MlLayer(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'max_position', **kwargs))
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.prefix = 'gpt2_ml'
    
    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        for i in range(self.num_hidden_layers):
            # qkv
            mapping = {
                f'h.{i}.attn.c_attn.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'h.{i}.attn.c_attn.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for old_key, new_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(old_key)) is None:
                    continue
                is_weight = old_key.endswith('weight')
                qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                    state_dict[new_key.format(i, i_k)] = i_v.T if is_weight else i_v
                state_dict.pop(old_key)
            
            # hdsz-hdsz的全连接
            old_key = f'h.{i}.attn.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.multiHeadAttention.o.weight'] = w.T
                state_dict.pop(old_key)

            # feed forward 第一层
            old_key = f'h.{i}.mlp.c_fc.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.intermediateDense.weight'] = w.T
                state_dict.pop(old_key)
                
            # feed forward 第二层
            old_key = f'h.{i}.mlp.c_proj.weight'
            if (w := state_dict.get(old_key)) is not None:
                state_dict[f'decoderLayer.{i}.feedForward.outputDense.weight'] = w.T
                state_dict.pop(old_key)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'wte.weight',
            'embeddings.position_embeddings.weight': 'wpe.weight',
            'embeddings.layerNorm.weight': 'emb_norm.weight',
            'embeddings.layerNorm.bias': 'emb_norm.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'h.{i}.attn.c_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'h.{i}.ln_1.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'h.{i}.ln_1.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'h.{i}.mlp.c_fc.bias',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'h.{i}.mlp.c_proj.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'h.{i}.ln_2.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'h.{i}.ln_2.bias'
            })
        return mapping

    class Gpt2MlLayer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        '''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # attn
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
            hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
            x = self.attnLayerNorm(hidden_states, conditional_emb)

            # ffn
            ffn_output = self.feedForward(x)
            # bert的第二个跳跃连接的输入1是经过了multiHeadAttention+attnLayerNorm的hidden_states, 即这里的x
            # gpt2_ml的第二个跳跃连接的输入1是经过了multiHeadAttention的hidden_states, 不加attnLayerNorm
            hidden_states = self.dropout_add(ffn_output, hidden_states)
            hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)

            if self.is_decoder and model_kwargs.get('use_states', False):
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs
