from typing import List, Optional, Tuple, Union
from bert4torch.models.qwen import Qwen2
from bert4torch.models.transformer import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict, inference_mode
import torch


class Qwen2VL(PreTrainedModelForDecoder):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
        vision_config = Qwen2VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(vision_config, attn_implementation=self.config._attn_implementation)
        self.model = Qwen2(**config)
        self.model.passed_kwargs = Qwen2VL.passed_kwargs

    def get_vllm_embedding(
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
        '''获取vlm的embedding
        1. train阶段：input_ids为query的token_ids, pixel_values或pixel_values_videos有一个不为空
        2. infer阶段：
            use_states=True:
                step=0: 和train阶段一致
                step=1: input_ids为新生成的last_token_id, pixel_values和pixel_values_videos为空
            use_states=False: 和train阶段一致
        '''
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

    def tie_weights(self):
        self.model.tie_weights()

    def forward(self, *inputs:Union[tuple, list], **model_kwargs):
        '''准备进embedding层的一些输入
        position_ids在之前已经准备好
        '''
        inputs = self.args_segmentate(inputs, **model_kwargs)
        input_ids, _, _, model_kwargs['attention_mask'], _, _, model_kwargs = self.model.preprare_embeddings_inputs(*inputs, **model_kwargs)
        inputs_embeds, model_kwargs['attention_mask'] = self.get_vllm_embedding(input_ids=input_ids, **model_kwargs)
        
        return self.model(input_ids=inputs_embeds, **model_kwargs)


    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'model.embeddings.word_embeddings.weight', 'model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'model.embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'model.lm_head.weight': 'lm_head.weight',
            'model.LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.model.num_hidden_layers):
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

    def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

            Explanation:
                Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

                For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
                Examples:
                    input_ids: [T T T T T], here T is for text.
                    temporal position_ids: [0, 1, 2, 3, 4]
                    height position_ids: [0, 1, 2, 3, 4]
                    width position_ids: [0, 1, 2, 3, 4]

                For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
                and 1D rotary position embeddin for text part.
                Examples:
                    Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                    input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                    vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                    vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                    vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                    text temporal position_ids: [3, 4, 5, 6, 7]
                    text height position_ids: [3, 4, 5, 6, 7]
                    text width position_ids: [3, 4, 5, 6, 7]
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
            video_token_id = self.config.video_token_id
            vision_start_token_id = self.config.vision_start_token_id
            mrope_position_deltas = []
            if image_grid_thw is not None or video_grid_thw is not None:
                total_input_ids = input_ids
                position_ids = torch.ones(
                    3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
                )
                image_index, video_index = 0, 0
                for i, input_ids in enumerate(total_input_ids):
                    if attention_mask is not None:
                        input_ids = input_ids[attention_mask[i] == 1]
                    image_nums, video_nums = 0, 0
                    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (vision_tokens == video_token_id).sum()
                    input_tokens = input_ids.tolist()
                    llm_pos_ids_list: list = []
                    st = 0
                    remain_images, remain_videos = image_nums, video_nums
                    for _ in range(image_nums + video_nums):
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if ed_image < ed_video:
                            t, h, w = (
                                image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2],
                            )
                            image_index += 1
                            remain_images -= 1
                            ed = ed_image
                        else:
                            t, h, w = (
                                video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2],
                            )
                            video_index += 1
                            remain_videos -= 1
                            ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if st < len(input_tokens):
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
                return position_ids, mrope_position_deltas
            else:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                    mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
                else:
                    position_ids = (
                        torch.arange(input_ids.shape[1], device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(3, input_ids.shape[0], -1)
                    )
                    mrope_position_deltas = torch.zeros(
                        [input_ids.shape[0], 1],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )

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
            # use_states且step>=1
            batch_size = input_ids.shape[0]
            cache_len = input_seqlen[0] + step - 1
            delta = (cache_len + rope_deltas if input_seqlen is not None and rope_deltas is not None else 0)
            position_ids = torch.arange(1, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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