from typing import List, Optional, Tuple, Union
from bert4torch.models.qwen import Qwen2
from bert4torch.models.base import PreTrainedModelForDecoder, register_model
from bert4torch.snippets import DottableDict
from .visual import GlmOcrVisionModel, GlmOcrVisionConfig
import torch
import itertools


@register_model(name="glm_ocr")
class GlmOcr(PreTrainedModelForDecoder):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        vision_config = GlmOcrVisionConfig.from_dict(self.config.vision_config)
        self.visual = GlmOcrVisionModel._from_config(vision_config)
        self.model = Qwen2(**self.config.text_config)
        self.model.passed_kwargs = GlmOcr.passed_kwargs

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """图片embedding"""
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        return image_embeds
    
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """视频embedding"""
        return self.get_image_features(pixel_values_videos, video_grid_thw)
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

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
        
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        return inputs_embeds, attention_mask

    def tie_weights(self):
        self.model.tie_weights()

    def forward(self, *inputs:Union[tuple, list], **model_kwargs):
        """准备进embedding层的一些输入
        position_ids在之前已经准备好
        """
        inputs = self.args_segmentate(inputs, **model_kwargs)
        input_ids, _, _, model_kwargs['attention_mask'], _, _, model_kwargs = self.model.preprare_embeddings_inputs(*inputs, **model_kwargs)
        inputs_embeds, model_kwargs['attention_mask'] = self.get_visual_embedding(input_ids=input_ids, **model_kwargs)
        
        return self.model(input_ids=inputs_embeds, **model_kwargs)


    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'model.embeddings.word_embeddings.weight', 'model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'model.embeddings.word_embeddings.weight': 'model.language_model.embed_tokens.weight',
            'model.lm_head.weight': 'lm_head.weight',
            'model.LayerNormFinal.weight': 'model.language_model.norm.weight',
            }

        for i in range(self.model.num_hidden_layers):
            mapping.update( 
            {
                f'model.decoderLayer.{i}.multiHeadAttention.q.weight': f'model.language_model.layers.{i}.self_attn.q_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.q.bias': f'model.language_model.layers.{i}.self_attn.q_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.k.weight': f'model.language_model.layers.{i}.self_attn.k_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.k.bias': f'model.language_model.layers.{i}.self_attn.k_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.v.weight': f'model.language_model.layers.{i}.self_attn.v_proj.weight',
                f'model.decoderLayer.{i}.multiHeadAttention.v.bias': f'model.language_model.layers.{i}.self_attn.v_proj.bias',
                f'model.decoderLayer.{i}.multiHeadAttention.o.weight': f'model.language_model.layers.{i}.self_attn.o_proj.weight',
                f'model.decoderLayer.{i}.attnLayerNorm.weight': f'model.language_model.layers.{i}.input_layernorm.weight',
                f'model.decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.language_model.layers.{i}.mlp.gate_up_proj.weight',
                f'model.decoderLayer.{i}.feedForward.outputDense.weight': f'model.language_model.layers.{i}.mlp.down_proj.weight',
                f'model.decoderLayer.{i}.ffnLayerNorm.weight': f'model.language_model.layers.{i}.post_attention_layernorm.weight'
            })

        for model_key, _ in self.visual.named_parameters():
            mapping[f'visual.{model_key}'] = f'model.visual.{model_key}'
        return mapping

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id

        mrope_position_deltas = []
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        video_group_index = 0
        image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
        video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            input_tokens = input_ids.tolist()

            input_token_type = []
            video_check_flg = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False
                if token == image_token_id and not video_check_flg:
                    input_token_type.append("image")
                elif token == image_token_id and video_check_flg:
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))
            llm_pos_ids_list = []
            video_frame_num = 1
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if modality_type == "image":
                    t, h, w = image_grid_thw_list[image_index]
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                    image_index += 1
                    video_frame_num = 1
                elif modality_type == "video":
                    _, h, w = video_grid_thw_list[video_index]
                    t = video_frame_num
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    for t_idx in range(llm_grid_t):
                        t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                    video_group_index += 1
                    if video_group_index >= video_grid_thw_list[video_index][0]:
                        video_index += 1
                        video_group_index = 0
                    video_frame_num += 1
                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    video_frame_num = 1
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        step=None,
        input_seqlen=None,
        output_ids=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        use_states=False,
        **kwargs,
    ):
        # 这里主要是需要处理下position_ids和rope_deltas
        rope_deltas = kwargs.get("rope_deltas", None)
        if step == 0 or (use_states is False):
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
        else:
            position_ids = rope_deltas
            # 无需重新生成vlm embedding
            pixel_values = None
            pixel_values_videos = None

        kwargs.update(
            {   
                "use_states": use_states,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return kwargs