from bert4torch.models.llama import LLaMA
from ..base import register_model


@register_model(name="minicpm")
class MiniCPM(LLaMA):
    _no_split_modules = ["MiniCPMLayer"]
    def __init__(self, *args, **kwargs):
        # kwargs['layer_type'] = 'MiniCPMLayer'
        super().__init__(*args, **kwargs)
        self.logit_scale = 1 / (self.hidden_size / kwargs.get('dim_model_base'))

    @property
    def _layer_args(self):
        args = super()._layer_args
        return args + ['num_hidden_layers']