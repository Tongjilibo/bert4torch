from typing import List, Optional
from bert4torch.models.qwen import Qwen2
from bert4torch.models.base import BERT_BASE
from bert4torch.snippets import DottableDict, inference_mode
import torch


class Qwen2VL(BERT_BASE):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
        vision_config = Qwen2VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(vision_config, attn_implementation=self.config._attn_implementation)
        self.model = Qwen2(**config)

    def tie_weights(self):
        self.model.tie_weights()

    def get_vllm_embedding(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
        ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embeddings(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
            return inputs_embeds, attention_mask

    def forward(
            self,         
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
    ):

        inputs_embeds, attention_mask = self.get_vllm_embedding(
            input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)

        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

    def load_variable(self, variable, old_key, new_key):
        if old_key in {'model.embeddings.word_embeddings.weight', 'model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'model.embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'model.lm_head.weight': 'lm_head.weight',
            'model.LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'model.decoderLayer.{i}.multiHeadAttention.q.weight': f'model.layers.{i}.self_attn.q_proj.weight',
            f'model.decoderLayer.{i}.multiHeadAttention.q.bias': f'model.layers.{i}.self_attn.q_proj.bias',
            f'model.decoderLayer.{i}.multiHeadAttention.k.weight': f'model.layers.{i}.self_attn.k_proj.weight',
            f'model.decoderLayer.{i}.multiHeadAttention.k.bias': f'model.layers.{i}.self_attn.k_proj.bias',
            f'model.decoderLayer.{i}.multiHeadAttention.v.weight': f'model.layers.{i}.self_attn.v_proj.weight',
            f'model.decoderLayer.{i}.multiHeadAttention.v.bias': f'model.layers.{i}.self_attn.v_proj.bias',
            f'model.decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.self_attn.o_proj.weight',
            f'model.decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'model.decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
            f'model.decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
            f'model.decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'model.decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping

    
    def _decode_stream(self, inputs_embeds, attention_mask, **kwargs):
        for output in self.model.stream_generate(
            inputs_embeds, attention_mask=attention_mask, **kwargs):
            yield output

    @inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_vllm_embedding(
            input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)

        return self.model.generate(inputs_embeds, attention_mask=attention_mask)
