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
        self.visual = Qwen3VLVisionModel._from_config(vision_config, attn_implementation=self.config._attn_implementation)
        self.model = Qwen3(**self.config.text_config)
        self.model.passed_kwargs = Qwen3VL.passed_kwargs