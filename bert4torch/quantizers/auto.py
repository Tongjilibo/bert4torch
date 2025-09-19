from .quantizer_awq import AwqQuantizer
from .quantizer_gptq import GptqQuantizer
from .quantizer_bnb_kbit import BnbkBitHfQuantizer
from .quantizer_cpm_kernels import CpmKernelQuantizer

AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "load_in_8bit": BnbkBitHfQuantizer,
    "load_in_4bit": BnbkBitHfQuantizer,
    "gptq": GptqQuantizer,
    "cpm_kernels": CpmKernelQuantizer
}