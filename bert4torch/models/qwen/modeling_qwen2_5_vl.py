from bert4torch.models.qwen import Qwen2
from bert4torch.models.qwen.modeling_qwen2_vl import Qwen2VL
from bert4torch.models.base import PreTrainedModelForDecoder, register_model
from bert4torch.snippets import DottableDict


@register_model(name="qwen2_5_vl")
class Qwen2_5VL(Qwen2VL):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        PreTrainedModelForDecoder.__init__(self, **config)
        self.config = DottableDict(config)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        vision_config = Qwen2_5_VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision_config)
        self.model = Qwen2(**config)
        self.model.passed_kwargs = Qwen2_5VL.passed_kwargs