import torch.nn as nn
from .activations import get_activation


class BottleneckAdapterLayer(nn.Module):
    """
    The adapters first project the original d-dimensional features into a smaller dimension, m,
    apply a nonlinearity, then project back to d dimensions

    """

    def __init__(self,
                 adapter_input_size,
                 bottleneck_size,
                 adapter_non_linearity='gelu'):
        super().__init__()

        self.adapter_input_size = adapter_input_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = get_activation(adapter_non_linearity)

        # down proj
        self.down_proj = nn.Linear(self.adapter_input_size, self.bottleneck_size)
        # up proj
        self.up_proj = nn.Linear(self.bottleneck_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self, init_mean=0.0, init_std=0.01):
        self.down_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output
