from .modeling_qwen3 import Qwen3
from bert4torch.layers.core import Qwen3MoeSparseFeedForward
import re
from ..base import register_model


@register_model(name="qwen3_moe")
class Qwen3Moe(Qwen3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_only_layers = kwargs.get('mlp_only_layers')
        self.num_experts = kwargs.get('num_experts')
        self.decoder_sparse_step = kwargs.get('decoder_sparse_step')
        self.num_experts = kwargs.get('num_experts')
        for layer_idx, layer in enumerate(self.decoderLayer):
            if (layer_idx not in self.mlp_only_layers) and (
                self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0
            ):
                layer.feedForward = Qwen3MoeSparseFeedForward(**kwargs)
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        mapping = {k:v for k,v in mapping.items() if not re.search('decoderLayer\\.[0-9]+\\.feedForward', k)}

        for i in range(self.num_hidden_layers):
            if (i not in self.mlp_only_layers) and (
                self.num_experts > 0 and (i + 1) % self.decoder_sparse_step == 0
            ):
                mapping[f'decoderLayer.{i}.feedForward.gate.weight'] = f"model.layers.{i}.mlp.gate.weight"
                for j in range(self.num_experts):
                    mapping.update({
                    f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense.weight': f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight',
                    f'decoderLayer.{i}.feedForward.experts.{j}.intermediateDense2.weight': f'model.layers.{i}.mlp.experts.{j}.up_proj.weight',
                    f'decoderLayer.{i}.feedForward.experts.{j}.outputDense.weight': f'model.layers.{i}.mlp.experts.{j}.down_proj.weight',
                    })
        return mapping