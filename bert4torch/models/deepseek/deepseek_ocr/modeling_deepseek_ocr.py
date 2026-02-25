'''DeepseekOCR: https://github.com/deepseek-ai/DeepSeek-OCR
image_encoder: sam+clip
language_model: deepseekv2
'''
from ..modeling_deepseek_v2 import DeepSeekV2
from bert4torch.models.base import PreTrainedModelForDecoder, register_model
from .deepencoder_common import build_sam_vit_b, MlpProjector
from .deepencoder_clip import build_clip_l
import torch
from torch import nn
from bert4torch.snippets import DottableDict
from typing import Union, Optional


@register_model(name="deepseek_ocr")
class DeepSeekOCR(PreTrainedModelForDecoder):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"images_ori", "images_crop", "images_seq_mask", "images_spatial_crop"}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = DottableDict(kwargs)

        self.sam_model = build_sam_vit_b(**self.config.vision_config.width.sam_vit_b)
        self.vision_model = build_clip_l()
        n_embed = 1280
        self.projector =  MlpProjector(DottableDict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
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
                        local_features_2 = self.vision_model(patches, local_features_1)  
                        local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        local_features = self.projector(local_features)

                        global_features_1 = self.sam_model(image_ori)
                        global_features_2 = self.vision_model(image_ori, global_features_1) 
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        global_features = self.projector(global_features)

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)

                        _2, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2 ** 0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat([global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
                        global_features = global_features.view(-1, n_dim)

                        local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                        local_features = torch.cat([local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1)
                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                   
                    else:
                        global_features_1 = self.sam_model(image_ori)
                        global_features_2 = self.vision_model(image_ori, global_features_1) 
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        global_features = self.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat([global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
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
            'sam_model': self.sam_model.named_parameters(),
            'vision_model': self.vision_model.named_parameters(),
            'projector': self.projector.named_parameters()
        }
        for name, named_parameters in name_module.items():
            new_mapping.update({f'{name}.{model_key}':f'model.{name}.{model_key}' for model_key, _ in named_parameters})
        new_mapping[f'view_seperator'] = f'model.view_seperator'
        new_mapping[f'image_newline'] = f'model.image_newline'
        return new_mapping