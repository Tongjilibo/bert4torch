'''DeepseekOCR2: https://github.com/deepseek-ai/DeepSeek-OCR-2
image_encoder: sam+qwen2
language_model: deepseekv2
'''
from ..modeling_deepseek_v2 import DeepSeekV2
from bert4torch.models.base import PreTrainedModelForDecoder, register_model
from ..deepseek_ocr.deepencoder_common import build_sam_vit_b, MlpProjector
from .deepencoderv2_qwen2 import build_qwen2_decoder_as_encoder
import torch
from torch import nn
from bert4torch.snippets import DottableDict
from typing import Union, Optional


@register_model(name="deepseek_ocr2")
class DeepSeekOCR2(PreTrainedModelForDecoder):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"images_ori", "images_crop", "images_seq_mask", "images_spatial_crop"}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = DottableDict(kwargs)

        self.sam_model = build_sam_vit_b(**self.config.vision_config.width.sam_vit_b)
        self.vision_model = build_qwen2_decoder_as_encoder()
        n_embed = 1280
        self.projector =  MlpProjector(DottableDict(projector_type="linear", input_dim=896, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        # self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.model = DeepSeekV2(**self.config.language_config)

    def load_variable(self, variable, ckpt_key, model_key):
        if ckpt_key in {'model.embeddings.word_embeddings.weight', 'model.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable

    def tie_weights(self):
        self.model.tie_weights()

    def get_visual_embedding(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            images_ori: Optional[torch.FloatTensor] = None,
            images_crop: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.FloatTensor] = None,
            images_spatial_crop: Optional[torch.FloatTensor] = None,
            **kwargs
        ):
        """获取visual的embedding
        """
        if inputs_embeds is None:
            inputs_embeds = self.model.embeddings(input_ids)

        images = [(images_crop.to(self.device), images_ori.to(self.device))]
        if self.sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:
            idx = 0

            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]

                with torch.no_grad():                    
                    if torch.sum(patches).item() != 0:
                        # P, C, H, W = patches.shape
                        local_features_1 = self.sam_model(patches)

                        local_features_2 = self.vision_model(local_features_1)  
                        local_features = local_features_2
                        local_features = self.projector(local_features)

                        global_features_1 = self.sam_model(image_ori)
                        global_features_2 = self.vision_model(global_features_1) 
                        global_features = global_features_2
                        global_features = self.projector(global_features)

                        _, hw, n_dim = global_features.shape
                        _, hw2, n_dim2 = local_features.shape

                        global_features = global_features.view(-1, n_dim)
                        local_features = local_features.view(-1, n_dim2)
                        global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)

                    else:
                        global_features_1 = self.sam_model(image_ori)
                        global_features_2 = self.vision_model(global_features_1) 
                        global_features = global_features_2
                        global_features = self.projector(global_features)
                        _, hw, n_dim = global_features.shape

                        global_features = global_features.view(-1, n_dim)
                        global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)

                idx += 1
        return inputs_embeds

    def forward(self, *inputs:Union[tuple, list], **model_kwargs):
        """准备进embedding层的一些输入
        position_ids在之前已经准备好
        """
        inputs = self.args_segmentate(inputs, **model_kwargs)
        input_ids, _, _, model_kwargs['attention_mask'], _, _, model_kwargs = self.model.preprare_embeddings_inputs(*inputs, **model_kwargs)
        inputs_embeds = self.get_visual_embedding(input_ids=input_ids, **model_kwargs)
        
        return self.model(input_ids=inputs_embeds, **model_kwargs)

    def variable_mapping(self):
        # 映射到权重格式
        new_mapping = {'model.'+new_key: old_key for new_key, old_key in self.model.variable_mapping().items()}
        name_module = {
            ('sam_model', 'sam_model'): self.sam_model.named_parameters(),
            ('qwen2_model', 'vision_model'): self.vision_model.named_parameters(),
            ('projector', 'projector'): self.projector.named_parameters()
        }
        for (old_name, new_name), named_parameters in name_module.items():
            new_mapping.update({f'{new_name}.{model_key}':f'model.{old_name}.{model_key}' for model_key, _ in named_parameters})
        new_mapping[f'view_seperator'] = f'model.view_seperator'
        return new_mapping