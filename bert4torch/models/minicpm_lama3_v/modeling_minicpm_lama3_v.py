from bert4torch.models.llama import LLaMA
from ..base import register_model
from ..minicpmv import MiniCPMV
from bert4torch.snippets import DotDict
import inspect
from .modeling_idefics2_visual import Idefics2VisionTransformer

@register_model(name="minicpm_llama3_v")
class MiniCPMLlama3V(MiniCPMV):
    def __init__(self, **config):
        super().__init__(**config)
        self.llm = LLaMA(**config)
        self.config = DotDict(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.hidden_size

    def init_vision_module(self):
        # from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
        # from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig
        # vision_config = Idefics2VisionConfig(**self.config.vision_config)
        model = Idefics2VisionTransformer(DotDict(self.config.vision_config))
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)
        self.vlm_tgt_sizes = True if 'tgt_sizes' in inspect.signature(model).parameters else False

        return model
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        mapping = {k:v for k,v in mapping.items() if '.bias' not in k}
        return mapping