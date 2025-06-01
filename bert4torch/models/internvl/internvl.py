from typing import List, Optional, Tuple, Union
from bert4torch.models.qwen import Qwen2
from bert4torch.models.llama import LLaMA
from bert4torch.models.transformer import PreTrainedModelForDecoder
from bert4torch.snippets import DottableDict, inference_mode, log_warn_once
import torch
from torch import nn


class InternVL(PreTrainedModelForDecoder):
    _no_split_modules = ['InternVisionModel', 'BertLayer']
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        self.select_layer = self.config.select_layer
        self.downsample_ratio = self.config.downsample_ratio
        use_flash_attn = has_flash_attn if has_flash_attn else False
        self.config.vision_config.use_flash_attn = True if use_flash_attn else False
        self.config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        self.ps_version = self.config.ps_version
        self.img_context_token_id = self.config.img_context_token_id

        # 模型结构
        from .modeling_intern_vit import InternVisionModel, has_flash_attn
        self.vision_model = InternVisionModel(self.config.vision_config)
        if self.config.model_llm == 'llama':
            self.language_model = LLaMA(**self.config)
        elif self.config.model_llm == 'qwen2':
            self.language_model = Qwen2(**self.config)
        else:
            raise NotImplementedError(f'{self.config.model_llm} is not implemented.')

        self.language_model.passed_kwargs = InternVL.passed_kwargs

        vit_hidden_size = self.config.vision_config.hidden_size
        llm_hidden_size = self.config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            log_warn_once("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def get_vllm_embedding(
            self, 
            input_ids: torch.LongTensor = None,
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
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
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
            vit_embeds:torch.Tensor = vit_embeds[:, 1:, :]

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
        return input_embeds

    def tie_weights(self):
        self.language_model.tie_weights()

    def load_variable(self, *args, **kwargs):
        return self.language_model.load_variable(*args, **kwargs)

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