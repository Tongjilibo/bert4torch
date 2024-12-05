'''多模态llama
'''
from typing import List, Optional, Tuple, Union
from bert4torch.layers import MllamaCrossAttentionDecoderLayer
from bert4torch.models.llama import LLaMA
from bert4torch.models.transformer import DecoderBase
from bert4torch.snippets import DottableDict, inference_mode
from torch import nn

__all__ = ['Mllama']

class MllamaTextModel(LLaMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer_idx in range(self.num_hidden_layers):
            if layer_idx in kwargs['cross_attention_layers']:
                self.decoderLayer[layer_idx] = MllamaCrossAttentionDecoderLayer(layer_idx=layer_idx, **self.get_kw(*self._layer_args, **kwargs))


class Mllama(DecoderBase):
    passed_kwargs = DecoderBase.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        from transformers.models.mllama.modeling_mllama import MllamaVisionModel
        from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig
        vision_config = MllamaVisionConfig.from_dict(self.config.vision_config)
        self.visual = MllamaVisionModel._from_config(vision_config)
        self.language_model = MllamaTextModel(**config)
        self.multi_modal_projector = nn.Linear(
            self.config.vision_config.vision_output_dim,
            self.config.text_config.hidden_size,
            bias=True,
        )
        self.language_model.passed_kwargs = Mllama.passed_kwargs


    def load_variable(self, variable, old_key, new_key):
        if old_key in {'language_model.embeddings.word_embeddings.weight', 'language_model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'language_model.embeddings.word_embeddings.weight': 'language_model.model.embed_tokens.weight',
            'language_model.lm_head.weight': 'language_model.lm_head.weight',
            'language_model.LayerNormFinal.weight': 'language_model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'language_model.decoderLayer.{i}.multiHeadAttention.q.weight': f'language_model.model.layers.{i}.self_attn.q_proj.weight',
            f'language_model.decoderLayer.{i}.multiHeadAttention.k.weight': f'language_model.model.layers.{i}.self_attn.k_proj.weight',
            f'language_model.decoderLayer.{i}.multiHeadAttention.v.weight': f'language_model.model.layers.{i}.self_attn.v_proj.weight',
            f'language_model.decoderLayer.{i}.multiHeadAttention.o.weight': f'language_model.model.layers.{i}.self_attn.o_proj.weight',
            f'language_model.decoderLayer.{i}.attnLayerNorm.weight': f'language_model.model.layers.{i}.input_layernorm.weight',
            f'language_model.decoderLayer.{i}.feedForward.intermediateDense.weight': f'language_model.model.layers.{i}.mlp.gate_proj.weight',
            f'language_model.decoderLayer.{i}.feedForward.intermediateDense2.weight': f'language_model.model.layers.{i}.mlp.up_proj.weight',
            f'language_model.decoderLayer.{i}.feedForward.outputDense.weight': f'language_model.model.layers.{i}.mlp.down_proj.weight',
            f'language_model.decoderLayer.{i}.ffnLayerNorm.weight': f'language_model.model.layers.{i}.post_attention_layernorm.weight'
            })
            if i in self.config.cross_attention_layers:
                mapping.update(
                {
                f'language_model.decoderLayer.{i}.crossAttention.k_norm.weight': f"language_model.model.layers.{i}.cross_attn.k_norm.weight",
                f'language_model.decoderLayer.{i}.crossAttention.k.weight': f"language_model.model.layers.{i}.cross_attn.k_proj.weight",
                f'language_model.decoderLayer.{i}.crossAttention.o.weight': f"language_model.model.layers.{i}.cross_attn.o_proj.weight",
                f'language_model.decoderLayer.{i}.crossAttention.q_norm.weight': f"language_model.model.layers.{i}.cross_attn.q_norm.weight",
                f'language_model.decoderLayer.{i}.crossAttention.q.weight': f"language_model.model.layers.{i}.cross_attn.q_proj.weight",
                f'language_model.decoderLayer.{i}.crossAttention.v.weight': f"language_model.model.layers.{i}.cross_attn.v_proj.weight",
                f'language_model.decoderLayer.{i}.crossAttention.cross_attn_attn_gate': f"language_model.model.layers.{i}.cross_attn_attn_gate",
                f'language_model.decoderLayer.{i}.crossAttention.cross_attn_mlp_gate': f"language_model.model.layers.{i}.cross_attn_mlp_gate",
                })
        return mapping