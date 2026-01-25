from bert4torch.models.qwen import Qwen3
from bert4torch.models.qwen.modeling_qwen2_vl import Qwen2VL
from bert4torch.models.base import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict


class Qwen3VL(Qwen2VL):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        PreTrainedModelForDecoder.__init__(self, **config)
        self.config = DottableDict(config)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        vision_config = Qwen3VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen3VLVisionModel._from_config(vision_config)
        self.model = Qwen3(**self.config.text_config)
        self.model.passed_kwargs = Qwen3VL.passed_kwargs

    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'model.embeddings.word_embeddings.weight': 'model.language_model.embed_tokens.weight',
            'model.lm_head.weight': 'model.language_model.embed_tokens.weight',
            'model.LayerNormFinal.weight': 'model.language_model.norm.weight',
            }

        for i in range(self.model.num_hidden_layers):
            mapping.update({
                f'model.decoderLayer.{i}.multiHeadAttention.q.weight': f'model.language_model.layers.{i}.self_attn.q_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.q.bias': f'model.language_model.layers.{i}.self_attn.q_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.k.weight': f'model.language_model.layers.{i}.self_attn.k_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.k.bias': f'model.language_model.layers.{i}.self_attn.k_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.v.weight': f'model.language_model.layers.{i}.self_attn.v_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.v.bias': f'model.language_model.layers.{i}.self_attn.v_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.o.weight': f'model.language_model.layers.{i}.self_attn.o_proj.weight',
                f'model.decoderLayer.{i}.attnLayerNorm.weight': f'model.language_model.layers.{i}.input_layernorm.weight',
                f'model.decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.language_model.layers.{i}.mlp.gate_proj.weight',
                f'model.decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.language_model.layers.{i}.mlp.up_proj.weight',
                f'model.decoderLayer.{i}.feedForward.outputDense.weight': f'model.language_model.layers.{i}.mlp.down_proj.weight',
                f'model.decoderLayer.{i}.ffnLayerNorm.weight': f'model.language_model.layers.{i}.post_attention_layernorm.weight'
            })
        
        for model_key, _ in self.visual.named_parameters():
            mapping[f'visual.{model_key}'] = f'model.visual.{model_key}'

        return mapping