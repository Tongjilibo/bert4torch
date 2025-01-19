from typing import List, Optional, Tuple, Union
from bert4torch.models.qwen import Qwen2
from bert4torch.models.llama import LLaMA
from bert4torch.models.transformer import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict, inference_mode
import torch
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from torch import nn


class InternVL(PreTrainedModelForDecoder):
    _no_split_modules = ['InternVisionModel', 'BertLayer']
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = self.config.select_layer
        self.template = self.config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))
        self.downsample_ratio = self.config.downsample_ratio
        self.ps_version = self.config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        self.config.vision_config.use_flash_attn = True if use_flash_attn else False
        self.config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        self.vision_model = InternVisionModel(self.config.vision_config)

        if self.config.llm_config.model == 'llama':
            self.language_model = LLaMA(**self.config.llm_config)
        elif self.config.llm_config.model == 'qwen2':
            self.language_model = Qwen2(**self.config.llm_config)
        else:
            raise NotImplementedError(f'{self.config.llm_config.architectures[0]} is not implemented.')

        self.language_model.passed_kwargs = InternVL.passed_kwargs

        vit_hidden_size = self.config.vision_config.hidden_size
        llm_hidden_size = self.config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

    def get_vllm_embedding(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
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
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.vision_model.get_dtype())

                if self.select_layer == -1:
                    vit_embeds = self.vision_model(
                        pixel_values=pixel_values,
                        output_hidden_states=False,
                        return_dict=True).last_hidden_state
                else:
                    vit_embeds = self.vision_model(
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        return_dict=True).hidden_states[self.select_layer]
                vit_embeds = vit_embeds[:, 1:, :]

                h = w = int(vit_embeds.shape[1] ** 0.5)
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
                vit_embeds = self.mlp1(vit_embeds)

                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                assert selected.sum() != 0
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds = input_embeds.reshape(B, N, C)
        else:
            inputs_embeds = input_ids
        return inputs_embeds

    def tie_weights(self):
        self.language_model.tie_weights()

    def forward(self, *inputs:Union[tuple, list], **model_kwargs):
        '''准备进embedding层的一些输入
        '''
        inputs = self.args_segmentate(inputs, **model_kwargs)
        input_ids, _, _, _, _, _, model_kwargs = self.language_model.preprare_embeddings_inputs(*inputs, **model_kwargs)
        inputs_embeds = self.get_vllm_embedding(input_ids=input_ids, **model_kwargs)
        return self.language_model(input_ids=inputs_embeds, **model_kwargs)
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = self.language_model.variable_mapping()
        mapping = {f'language_model.{model_key}': f'language_model.{ckpt_key}' for model_key, ckpt_key in mapping.items()}
        return mapping
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        step=None,
        input_seqlen=None,
        output_ids=None,
        pixel_values=None,
        use_states=False,
        **kwargs,
    ):
        if step > 0 and use_states:
            pixel_values = None
        kwargs.update({"use_states": use_states, "pixel_values": pixel_values})
        return kwargs