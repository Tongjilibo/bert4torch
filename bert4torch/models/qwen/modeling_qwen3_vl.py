from bert4torch.models.qwen import Qwen3
from bert4torch.models.qwen.modeling_qwen2_vl import Qwen2VL
from bert4torch.models.base import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict
import torch
from typing import List, Optional, Tuple, Union


class Qwen3VLTextModel(Qwen3):
    def apply_on_layer_end(self, layer_idx, **model_kwargs):
        visual_pos_masks = model_kwargs.get("visual_pos_masks", None)
        deepstack_visual_embeds = model_kwargs.get("deepstack_visual_embeds", None)
        
        model_kwargs = super().apply_on_layer_end(layer_idx, **model_kwargs)
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            model_kwargs['hidden_states'] = self._deepstack_process(
                model_kwargs['hidden_states'],
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )
        return model_kwargs

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3VL(Qwen2VL):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        PreTrainedModelForDecoder.__init__(self, **config)
        self.config = DottableDict(config)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        vision_config = Qwen3VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen3VLVisionModel._from_config(vision_config)
        self.model = Qwen3VLTextModel(**self.config.text_config)
        self.model.passed_kwargs = Qwen3VL.passed_kwargs

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """图片embedding"""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_visual_embedding(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            **kwargs
        ):
        """获取visual的embedding
        1. train阶段：input_ids为query的token_ids, pixel_values或pixel_values_videos有一个不为空
        2. infer阶段：
            use_states=True:
                step=0: 和train阶段一致
                step=1: input_ids为新生成的last_token_id, pixel_values和pixel_values_videos为空
            use_states=False: 和train阶段一致
        """
        if inputs_embeds is None:
            inputs_embeds = self.model.embeddings(input_ids)
        
        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        return inputs_embeds, attention_mask, visual_pos_masks, deepstack_visual_embeds

    def forward(self, *inputs:Union[tuple, list], **model_kwargs):
        """准备进embedding层的一些输入
        position_ids在之前已经准备好
        """
        inputs = self.args_segmentate(inputs, **model_kwargs)
        input_ids, _, _, model_kwargs['attention_mask'], _, _, model_kwargs = self.model.preprare_embeddings_inputs(*inputs, **model_kwargs)
        inputs_embeds, model_kwargs['attention_mask'], visual_pos_masks, deepstack_visual_embeds = \
            self.get_visual_embedding(input_ids=input_ids, **model_kwargs)
        
        return self.model(
            input_ids=inputs_embeds, 
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **model_kwargs)


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
                f'model.decoderLayer.{i}.multiHeadAttention.q_norm.weight': f'model.language_model.layers.{i}.self_attn.q_norm.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.k.weight': f'model.language_model.layers.{i}.self_attn.k_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.k.bias': f'model.language_model.layers.{i}.self_attn.k_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.k_norm.weight': f'model.language_model.layers.{i}.self_attn.k_norm.weight',
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