'''多模态llama
'''
from typing import List, Optional, Tuple, Union
from bert4torch.models.llama import LLaMA
from bert4torch.models.transformer import DecoderBase
from bert4torch.snippets import DottableDict, inference_mode
import torch


class Mllama(DecoderBase):
    passed_kwargs = DecoderBase.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        from transformers.models.mllama.modeling_mllama import MllamaVisionModel
        from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig
        vision_config = MllamaVisionConfig.from_dict(self.config.vision_config)
        self.visual = MllamaVisionModel._from_config(vision_config)
        self.model = LLaMA(**config)
        self.model.passed_kwargs = Mllama.passed_kwargs
