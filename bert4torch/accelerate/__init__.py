"""
Copyed and simplified from HuggingFace accelerate
"""
from .utils.dataclasses import *
from .big_modeling import (
    dispatch_model,
    init_empty_weights,
    init_on_device,
)
from .utils import (
    infer_auto_device_map,
)
