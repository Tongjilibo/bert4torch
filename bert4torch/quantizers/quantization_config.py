import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
from packaging import version
from ..snippets import (
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    VPTQ = "vptq"
    QUANTO = "quanto"
    EETQ = "eetq"
    HIGGS = "higgs"
    HQQ = "hqq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FBGEMM_FP8 = "fbgemm_fp8"
    TORCHAO = "torchao"
    BITNET = "bitnet"
    SPQR = "spqr"
    FP8 = "fp8"
    QUARK = "quark"
    FPQUANT = "fp_quant"
    AUTOROUND = "auto-round"
    MXFP4 = "mxfp4"
    METAL = "metal"
    FOUR_OVER_SIX = "fouroversix"
    SINQ = "sinq"


class AwqFormat(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    LLM_AWQ = "llm-awq"


class AwqBackend(str, Enum):
    LEGACY_AWQ = "autoawq"
    AUTO = "auto"
    AUTO_TRAINABLE = "auto_trainable"
    MACHETE = "machete"
    MARLIN = "marlin"
    EXLLAMA_V2 = "exllama_v2"
    EXLLAMA_V1 = "exllama_v1"
    GEMM = "gemm"
    GEMM_TRITON = "gemm_triton"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    TORCH_AWQ = "torch_awq"
    TORCH_FUSED_AWQ = "torch_fused_awq"


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        yield from copy.deepcopy(self.__dict__).items()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Default behavior: no diffing implemented for this config.
        """
        return self.to_dict()

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_diff_dict() if use_diff else self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs



@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`list[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quantized 4-bit params.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs:
            logger.info(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for GPTQ quantization relying on the gptqmodel backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[list[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        act_group_aware (`bool`, *optional*, defaults to `True`):
            Use GAR (group aware activation order) during quantization. Has measurable positive impact on quantization
            quality. Only applicable when `desc_act = False`. Will forced to be `False` when `desc_act = True`.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symmetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        format (`str`, *optional*, defaults to `"gptq"`):
            GPTQ weight format. `gptq` (v1) is supported by gptqmodel. `gptq_v2` is gptqmodel only.
        meta (`dict[str, any]`, *optional*):
            Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
            i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
        backend (`str`, *optional*):
            Controls which kernel to use. Valid values for gptqmodel are `auto`, `auto_trainable` and more. Ref gptqmodel backends:
            https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`list[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`list[list[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    """

    def __init__(
        self,
        bits: int,
        tokenizer: Any = None,
        dataset: list[str] | str | None = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        act_group_aware: bool = True,
        sym: bool = True,
        true_sequential: bool = True,
        format: str = "gptq",
        meta: dict[str, Any] | None = None,
        backend: str | None = None,
        model_seqlen: int | None = None,
        block_name_to_quantize: str | None = None,
        module_name_preceding_first_block: list[str] | None = None,
        batch_size: int = 1,
        pad_token_id: int | None = None,
        max_input_length: int | None = None,
        cache_block_outputs: bool = True,
        modules_in_block_to_quantize: list[list[str]] | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.act_group_aware = act_group_aware
        self.sym = sym
        self.true_sequential = true_sequential
        self.format = format.lower()
        # Compatible with legacy field: checkpoint_format
        if kwargs.get("checkpoint_format") is not None:
            self.format = kwargs.pop("checkpoint_format").lower()
        self.meta = meta
        self.backend = backend.lower() if isinstance(backend, str) else backend
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.max_input_length = max_input_length
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.post_init()

    def get_loading_attributes(self):
        attributes_dict = copy.deepcopy(self.__dict__)
        loading_attributes = ["max_input_length", "backend"]
        loading_attributes_dict = {i: j for i, j in attributes_dict.items() if i in loading_attributes}
        return loading_attributes_dict

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.dataset is not None:
            if isinstance(self.dataset, str):
                if self.dataset not in ["wikitext2", "c4", "c4-new"]:
                    raise ValueError(
                        f"""You have entered a string value for dataset. You can only choose between
                        ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                    )
            elif not isinstance(self.dataset, list):
                raise ValueError(
                    f"""dataset needs to be either a list of string or a value in
                    ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                )

        # act_group_order is only applicable when `desc_act = False`
        if self.desc_act and self.act_group_aware:
            self.act_group_aware = False
            logger.warning("`act_group_aware` has been auto-disabled as it is not compatible with `desc_act = True`.")

        # make sure backend default stays consistent with gptqmodel expectations
        if self.backend is None:
            self.backend = "auto"
        if self.modules_in_block_to_quantize is not None:
            optimum_version = version.parse(importlib.metadata.version("optimum"))
            if optimum_version < version.parse("1.15.0"):
                raise ValueError(
                    "You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument, please upgrade `optimum` package to a version superior than 1.15.0 ."
                )

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        # Compatible with legacy field: checkpoint_format
        config_dict["checkpoint_format"] = self.format
        return config_dict

    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        """
        return self.to_dict()

    @classmethod
    def from_dict_optimum(cls, config_dict):
        """
        Get compatible class with optimum gptq config dict
        """

        config = cls(**config_dict)
        return config


@dataclass
class AwqConfig(GPTQConfig):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        backend (`AwqBackend`, *optional*, defaults to `AwqBackend.AUTO`):
            The quantization backend.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        backend: AwqBackend = AwqBackend.AUTO,
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        format = kwargs.pop("format", AwqFormat.GEMM)
        # Compatible with legacy field: version
        if kwargs.get("version") is not None:
            format = kwargs.pop("version").lower()
        # Compatible with legacy backend
        if backend == AwqBackend.LEGACY_AWQ:
            backend = AwqBackend.AUTO
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert

        super().__init__(bits=bits, group_size=group_size, backend=backend, format=format, **kwargs)
        self.quant_method = QuantizationMethod.AWQ

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.backend == "llm-awq":
            self.format = AwqFormat.LLM_AWQ
            self.backend = AwqBackend.AUTO

        if self.format not in AwqFormat.__members__.values():
            raise ValueError(f"Invalid format '{self.format}'. Must be one of: {[b.value for b in AwqFormat]}")

        if self.backend not in AwqBackend.__members__.values():
            raise ValueError(f"Invalid backend '{self.backend}'. Must be one of: {[b.value for b in AwqBackend]}")

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        config_dict.pop("checkpoint_format")
        # Compatible with legacy field: version
        config_dict["version"] = self.format
        return config_dict