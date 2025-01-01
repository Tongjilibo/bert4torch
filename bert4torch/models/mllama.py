'''多模态llama
'''
from typing import List, Optional, Tuple, Union
from bert4torch.layers import MllamaCrossAttentionDecoderLayer
from bert4torch.models.llama import LLaMA
from bert4torch.models.transformer import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict, inference_mode
from torch import nn
import torch

__all__ = ['Mllama']

class MllamaTextModel(LLaMA):
    '''Mllama的语音模型，主要区别是部分layer是cross_attention的'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs.update({'p_bias': 'rotary', 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 
                'is_decoder': True, 'final_layernorm': True, 'pre_layernorm': True, 
                'mlp_type': 'LlamaFeedForward'})
        for layer_idx in range(self.num_hidden_layers):
            if layer_idx in kwargs['cross_attention_layers']:
                self.decoderLayer[layer_idx] = MllamaCrossAttentionDecoderLayer(layer_idx=layer_idx, **self.get_kw(*self._layer_args, **kwargs))


class Mllama(PreTrainedModelForDecoder):
    _no_split_modules = [
        "MllamaVisionEncoderLayer",
        "MllamaCrossAttentionDecoderLayer",
        "BertLayer",
    ]
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'}
    def __init__(self, **config):
        self.config = DottableDict(config)
        super().__init__(**self.config)
        from transformers.models.mllama.modeling_mllama import MllamaVisionModel
        from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig
        vision_config = MllamaVisionConfig.from_dict(self.config.vision_config)
        self.vision_model = MllamaVisionModel._from_config(vision_config)
        self.language_model = MllamaTextModel(**self.config.text_config)
        # word_embedding部分是vocab_size+8
        self.language_model.embeddings.word_embeddings = nn.Embedding(self.config.text_config.vocab_size+8, self.config.text_config.hidden_size)

        self.multi_modal_projector = nn.Linear(
            self.config.vision_config.vision_output_dim,
            self.config.text_config.hidden_size,
            bias=True,
        )
        self.language_model.passed_kwargs = Mllama.passed_kwargs

    def forward(
            self, 
            *inputs:Union[tuple, list], 
            pixel_values=None,
            aspect_ratio_ids=None,
            aspect_ratio_mask=None,
            cross_attention_mask=None, 
            cache_position=None,
            **model_kwargs):
        inputs = self.args_segmentate(inputs, **model_kwargs)

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.config.text_config.hidden_size)
        else:
            cross_attention_states = None

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = self._prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        return self.language_model(input_ids=inputs[0], 
                                   cross_attention_states=cross_attention_states, 
                                   cross_attention_mask=cross_attention_mask,
                                   full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                                   **model_kwargs)
    
    def prepare_inputs_for_generation(
            self, 
            input_ids, 
            step=None,
            input_seqlen=None,
            output_ids=None,
            **model_kwargs):
        # TODO 增加cache_position的逻辑
        if step == 0 or (model_kwargs.get('use_states') is False):
            model_kwargs['cache_position'] = torch.arange(input_seqlen[0], device=input_ids.device)
        else:
            model_kwargs['cache_position'] = input_seqlen
        model_kwargs['cross_attention_mask_prev'] = model_kwargs.get('cross_attention_mask')
        return model_kwargs
    
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs:dict):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask_prev", None)

        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs)

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat([cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1)
        return model_kwargs

    @staticmethod
    def _prepare_cross_attention_mask(
        cross_attention_mask: torch.Tensor,
        num_vision_tokens: int,
        dtype: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # reshape so it can be used by attn module
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
        cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
        cross_attention_mask = cross_attention_mask.unsqueeze(1)

        # invert the mask
        inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
        cross_attention_mask = inverted_cross_attn_mask.masked_fill(
            inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
        )

        # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
        # last dimension contains negative infinity values, otherwise it's 1
        negative_inf_value = torch.finfo(dtype).min
        full_text_row_masked_out_mask = (
            (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        return cross_attention_mask, full_text_row_masked_out_mask


    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'language_model.embeddings.word_embeddings.weight', 'language_model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'language_model.embeddings.word_embeddings.weight': 'language_model.model.embed_tokens.weight',
            'language_model.lm_head.weight': 'language_model.lm_head.weight',
            'language_model.LayerNormFinal.weight': 'language_model.model.norm.weight',
            }

        for i in range(self.language_model.num_hidden_layers):
            if i in self.config.text_config.cross_attention_layers:
                mapping.update({
                    f'language_model.decoderLayer.{i}.crossAttention.k_norm.weight': f"language_model.model.layers.{i}.cross_attn.k_norm.weight",
                    f'language_model.decoderLayer.{i}.crossAttention.k.weight': f"language_model.model.layers.{i}.cross_attn.k_proj.weight",
                    f'language_model.decoderLayer.{i}.crossAttention.o.weight': f"language_model.model.layers.{i}.cross_attn.o_proj.weight",
                    f'language_model.decoderLayer.{i}.crossAttention.q_norm.weight': f"language_model.model.layers.{i}.cross_attn.q_norm.weight",
                    f'language_model.decoderLayer.{i}.crossAttention.q.weight': f"language_model.model.layers.{i}.cross_attn.q_proj.weight",
                    f'language_model.decoderLayer.{i}.crossAttention.v.weight': f"language_model.model.layers.{i}.cross_attn.v_proj.weight",
                    f'language_model.decoderLayer.{i}.cross_attn_attn_gate': f"language_model.model.layers.{i}.cross_attn_attn_gate",
                    f'language_model.decoderLayer.{i}.cross_attn_mlp_gate': f"language_model.model.layers.{i}.cross_attn_mlp_gate",
                })
            else:
                mapping.update({
                    f'language_model.decoderLayer.{i}.multiHeadAttention.q.weight': f'language_model.model.layers.{i}.self_attn.q_proj.weight',
                    f'language_model.decoderLayer.{i}.multiHeadAttention.k.weight': f'language_model.model.layers.{i}.self_attn.k_proj.weight',
                    f'language_model.decoderLayer.{i}.multiHeadAttention.v.weight': f'language_model.model.layers.{i}.self_attn.v_proj.weight',
                    f'language_model.decoderLayer.{i}.multiHeadAttention.o.weight': f'language_model.model.layers.{i}.self_attn.o_proj.weight'
                })
            mapping.update({
                f'language_model.decoderLayer.{i}.attnLayerNorm.weight': f'language_model.model.layers.{i}.input_layernorm.weight',
                f'language_model.decoderLayer.{i}.feedForward.intermediateDense.weight': f'language_model.model.layers.{i}.mlp.gate_proj.weight',
                f'language_model.decoderLayer.{i}.feedForward.intermediateDense2.weight': f'language_model.model.layers.{i}.mlp.up_proj.weight',
                f'language_model.decoderLayer.{i}.feedForward.outputDense.weight': f'language_model.model.layers.{i}.mlp.down_proj.weight',
                f'language_model.decoderLayer.{i}.ffnLayerNorm.weight': f'language_model.model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping