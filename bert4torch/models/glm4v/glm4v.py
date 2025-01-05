from bert4torch.models.glm import GLM2
from bert4torch.models.transformer import PreTrainedModelForDecoder
from .visual import EVA2CLIPModel
from bert4torch.snippets import DottableDict


class GLM4V(PreTrainedModelForDecoder):
    passed_kwargs = PreTrainedModelForDecoder.passed_kwargs | {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "rope_deltas"}
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        self.vision = EVA2CLIPModel(self.config)
        self.llm = GLM2(**config)

    def load_variable(self, *args, **kwargs):
        return self.llm.load_variable(*args, **kwargs)
    
    def load_trans_ckpt(self, state_dict):
        return self.llm.load_trans_ckpt(state_dict, prefix='llm.')

    def save_trans_ckpt(self):
        vision_state_dict = self.vision.state_dict()
        key_list = list(vision_state_dict.keys())
        for k in key_list:
            vision_state_dict[f'transformer.vision.{k}'] = vision_state_dict.pop(k)
            
        llm_state_dict = self.llm.save_trans_ckpt()
        return {**vision_state_dict, **llm_state_dict}
    
    def variable_mapping(self):
        mapping = self.llm.variable_mapping()
        new_mapping = dict()
        for model_key, ckpt_key in mapping.items():
            new_mapping[f'llm.{model_key}'] = ckpt_key
        
        for model_key, _ in self.vision.named_parameters():
            new_mapping[f'vision.{model_key}'] = f'transformer.vision.{model_key}'
        return new_mapping
    
    def load_embeddings(self, embeddings):
        return super().load_embeddings(embeddings)