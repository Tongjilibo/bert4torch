from torch4keras.model import BaseModel, BaseModelDP, BaseModelDDP
from torch4keras.trainer import Trainer
from bert4torch.models.albert import ALBERT, ALBERT_Unshared
from bert4torch.models.bart import BART
from bert4torch.models.base import PreTrainedModel, extend_with_base_model, extend_with_language_model, extend_with_unified_language_model
from bert4torch.models.bert import BERT
from bert4torch.models.deberta import DebertaV2
from bert4torch.models.electra import ELECTRA
from bert4torch.models.ernie import Ernie
from bert4torch.models.ernied4_5 import Ernied4_5
from bert4torch.models.gau_alpha import GAU_alpha
from bert4torch.models.modernbert import ModernBert
from bert4torch.models.glm import GLM, GLM2
from bert4torch.models.glm4v import GLM4V
from bert4torch.models.gpt import GPT, GPT2, GPT2_ML
from bert4torch.models.llama import LLaMA, Baichuan, MiniCPM
from bert4torch.models.mllama import Mllama
from bert4torch.models.minicpmv import MiniCPMV, MiniCPMLlama3V
from bert4torch.models.nezha import NEZHA
from bert4torch.models.roformer import RoFormer, RoFormerV2
from bert4torch.models.t5 import T5, T5_Encoder, T5_Decoder
from bert4torch.models.transformer import Transformer, Encoder, Decoder
from bert4torch.models.transformer_xl import Transformer_XL
from bert4torch.models.xlnet import XLNET
from bert4torch.models.uie import UIE
from bert4torch.models.bloom import Bloom
from bert4torch.models.qwen import Qwen, Qwen2, Qwen3, Qwen3Moe
from bert4torch.models.qwen2_vl import Qwen2VL
from bert4torch.models.internlm import InternLM, InternLM2
from bert4torch.models.internvl import InternVL
from bert4torch.models.falcon import Falcon
from bert4torch.models.deepseek import DeepSeek
from typing import Union, Literal
import json
import os
from bert4torch.models.modeling_utils import restore_default_torch_dtype, set_default_torch_dtype, get_device_map
from bert4torch.snippets import (
    log_warn_once, 
    log_error,
    is_flash_attn_available, 
    is_xformers_available, 
    is_torch_sdpa_available,
    is_accelerate_available,
    get_checkpoint_path, 
    get_config_path,
    DottableDict,
    has_meta_param
)

@restore_default_torch_dtype
def build_transformer_model(
        config_path: Union[str, os.PathLike] = None, 
        checkpoint_path: Union[str, os.PathLike, list] = None, 
        model: Union[str, PreTrainedModel] = None, 
        application: Literal['encoder', 'lm', 'unilm', None] = None, 
        add_trainer: bool = False, 
        verbose: int = 1, 
        **kwargs
        ) -> Union[PreTrainedModel, BERT, Transformer, Trainer]:
    """根据配置文件构建模型, 可选加载checkpoint权重, 类似AutoModel.from_pretrained(...)

    :param config_path: str, 模型的config文件地址, 大部分模型都提供了bert4torch_config.json
    :param checkpoint_path: str/list[str], 模型文件/文件夹地址, 默认值None表示不加载预训练模型
    :param model: str, 加载的模型结构, 这里Model也可以基于nn.Module自定义后传入, 默认为'bert'
    :param application: str, 模型应用, 支持encoder, lm和unilm格式, 默认为'encoder'
    :param segment_vocab_size: int, type_token_ids数量, 默认为2, 如不传入segment_ids则需设置为0
    :param with_pool: bool, 是否包含Pool部分, 默认为False
    :param with_nsp: bool, 是否包含NSP部分, 默认为False
    :param with_mlm: bool, 是否包含MLM部分, 默认为False
    :param output_all_encoded_layers: bool, 是否返回所有hidden_state层, 默认为False
    :param additional_embs: bool, 是否有额外的embedding输入
    :param keep_tokens: list[int], 精简词表, 保留的id的序号如：[0, 100, 101, 102, 106, 107, ...]
    :param pad_token_id: int, 默认为0, 部分模型padding不是0时在这里指定, 用于attention_mask生成, 如设置成-100
    :param custom_position_ids: bool, 是否自行传入位置id, True表示传入, False表示不传入, 'start_at_padding'表示从padding_idx+1开始, 默认为False
    :param custom_attention_mask: bool, 是否自行传入attention_mask, 默认为False
    :param shared_segment_embeddings: bool, 若True, 则segment跟token共用embedding, 默认为False
    :param conditional_size: conditional layer_norm, 默认为None
    :param compound_tokens: 扩展Embedding, 默认为None
    :param residual_attention_scores: bool, Attention矩阵加残差, 默认为False
    :param ignore_invalid_weights: bool, 允许跳过不存在的权重, 默认为False
    :param keep_hidden_layers: 保留的hidden_layer层的id, 默认为None表示全部使用
    :param hierarchical_position: 是否层次分解位置编码, 默认为None表示不使用
    :param gradient_checkpoint: bool, 是否使用gradient_checkpoint, 默认为False
    :param add_trainer: bool, 指定从BaseModel继承, 若build_transformer_model后需直接compile()、fit()需设置为True, 默认为None
    :param verbose: int, 是否显示加载权重的[WARNING]信息, 默认为1表示显示未加载的, 2表示显示所有不匹配的, 0表示不显示

    > 大模型参数
    :param skip_init/low_cpu_mem_usage: bool, 是否初始化, 默认为False
    :param device_map: None/str/dict, 为不同Module指定不同的device, 默认为None表示加载到cpu中, 不同于transformer自动分配, 这里需手动指定dict
    :param torch_dtype: 指定权重的dtype
    :param flash_attention: bool/str, 是否使用flash_attention, 默认为None
    :param use_logn_attn: bool, 在attention模块中是否使用logn_attn
    :param num_key_value_heads: int, 使用MQA的头数
    :param ntk_alpha: float, rope外推使用ntk方法时的alhpa参数

    :return: A pytorch model instance

    Examples(支持几种加载方式):
    ```python
    >>> # 1. 仅指定config_path: 从头初始化模型结构, 不加载预训练模型
    >>> model = build_transformer_model('./model/bert4torch_config.json')

    >>> # 2. 仅指定checkpoint_path: 
    >>> # 2.1 文件夹路径: 自动寻找路径下的*.bin/*.safetensors权重文件 + bert4torch_config.json/config.json文件
    >>> model = build_transformer_model(checkpoint_path='./model')

    >>> # 2.2 文件路径/列表: 文件路径即权重路径/列表, config会从同级目录下寻找
    >>> model = build_transformer_model(checkpoint_path='./pytorch_model.bin')

    >>> # 2.3 model_name: hf上预训练权重名称, 会自动下载hf权重以及bert4torch_config.json文件
    >>> model = build_transformer_model(checkpoint_path='google-bert/bert-base-chinese')

    >>> # 3. 同时指定config_path和checkpoint_path(本地路径名或model_name排列组合): 
    >>> config_path = './model/bert4torch_config.json'  # 或'google-bert/bert-base-chinese'
    >>> checkpoint_path = './model/pytorch_model.bin'  # 或'google-bert/bert-base-chinese'
    >>> model = build_transformer_model(config_path, checkpoint_path)
    ```
    """
    # 校验checkpoint_path, config_path
    config_path = get_config_path(config_path if config_path is not None else checkpoint_path, **kwargs)
    checkpoint_path = get_checkpoint_path(checkpoint_path, **kwargs)
    if (config_path is None) and (checkpoint_path is not None):
        # 没有找到bert4torch_config.json，则从local的checkpoint_path去找
        config_path = get_config_path(checkpoint_path, **kwargs)

    # config的修改
    config = check_update_config(config_path, **kwargs)
    config['add_trainer'] = add_trainer

    device_map = config.pop('device_map', None)
    skip_init = config.pop('skip_init', False) or config.pop('low_cpu_mem_usage', False)
    skip_init = True if device_map is not None else skip_init  # 指定了device_map, 就必须skip_init
    torch_dtype = config.pop('torch_dtype', None)
    checkpoint_path = checkpoint_path or config.get('checkpoint_path')

    models = {
        'bert': BERT,
        'roberta': BERT,  
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'nezha': NEZHA,
        'roformer': RoFormer,
        'roformer_v2': RoFormerV2,
        'gau_alpha': GAU_alpha,
        'electra': ELECTRA,
        'ernie': Ernie,
        'ernie4_5': Ernied4_5,
        'deberta_v2': DebertaV2,
        'modernbert': ModernBert,
        'uie': UIE,
        'encoder': Encoder,
        'decoder': Decoder,
        'transformer': Transformer,
        'bart': BART,
        'gpt': GPT,
        'gpt2': GPT2,
        'gpt2_ml': GPT2_ML,
        'llama': LLaMA,
        'mllama': Mllama,
        'baichuan': Baichuan,
        'glm': GLM,
        'chatglm': GLM,
        'glm2': GLM2,
        'chatglm2': GLM2,
        'glm4v': GLM4V,
        't5': T5,
        't5_encoder': T5_Encoder,
        't5_decoder': T5_Decoder,
        't5.1.0': T5,
        't5.1.0_encoder': T5_Encoder,
        't5.1.0_decoder': T5_Decoder,
        't5.1.1': T5,
        't5.1.1_encoder': T5_Encoder,
        't5.1.1_decoder': T5_Decoder,
        'mt5.1.1': T5,
        'mt5.1.1_encoder': T5_Encoder,
        'mt5.1.1_decoder': T5_Decoder,
        'transformer_xl': Transformer_XL,
        'xlnet': XLNET,
        'bloom': Bloom,
        'qwen': Qwen,
        'qwen2': Qwen2,
        'qwen3': Qwen3,
        'qwen3_moe': Qwen3Moe,
        'qwen2_vl': Qwen2VL,
        'internlm': InternLM,
        'internlm2': InternLM2,
        'internvl': InternVL,
        'falcon': Falcon,
        'deepseek': DeepSeek,
        'minicpm': MiniCPM,
        'minicpmv': MiniCPMV,
        'minicpm_llama3_v': MiniCPMLlama3V
    }

    model = model or config.get('model', config.get('model_type', 'bert'))
    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            config['version'] = model
    elif isinstance(model, type) and issubclass(model, PreTrainedModel): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('Args `model` type should be string or PreTrainedModel')

    # 使用 lm/unilm
    application = (application or config.get('application', 'encoder')).lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5', ]:
        raise ValueError(f'"{model}" model can not be used as "{application}" application.\n')
    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    # 动态继承BaseModel, 直接加载预训练模型训练时使用
    if add_trainer:
        MODEL = extend_with_base_model(MODEL)

    # 指定默认权重类型
    if torch_dtype is not None:
        torch_dtype = set_default_torch_dtype(torch_dtype, model, config)

    # 生成网络结构
    if skip_init and (checkpoint_path is not None):
        if is_accelerate_available():
            from accelerate import init_empty_weights
            with init_empty_weights():
                transformer = MODEL(**config)
        else:
            skip_init = False  # 若accelerate包不存在则先初始化模型参数
            log_warn_once('Package `accelerate` not available, use `pip install accelerate`')
    if not skip_init:
        transformer = MODEL(**config)
        transformer.apply(transformer.init_model_weights)  # 初始化权重

    transformer.config = config
    # 预训练模型是否已量化, 加载量化后的权重使用, 如果是加载原权重再自行量化这里不需要设置
    pre_quantized = hasattr(config, "quantization_config")
    if pre_quantized:
        transformer = transformer.quantize(device_map=device_map, torch_dtype=torch_dtype, 
                                           **config.pop('quantization_config'))
    
    # 权重加载
    transformer.checkpoint_path = checkpoint_path
    if checkpoint_path is not None:
        # 根据模型尺寸和硬件(gpu, cpu)的大小来确定device_map
        device_map = get_device_map(transformer, device_map, torch_dtype, **config)
        transformer.from_pretrained(checkpoint_path, mapping=config.pop('mapping', None), 
                                    device_map=device_map, torch_dtype=torch_dtype, verbose=verbose, **config)
    
    # 权重tie, 若skip_init则模型结构中的tie_weights会失效, 这里重新tie_weights一下
    transformer.tie_weights()

    # meta device则报错
    if device_map is None:
        has_meta_param(transformer, verbose=True)

    if hasattr(transformer, 'quantizer'):
        transformer.quantizer.postprocess_model(transformer, config=config)

    return transformer


def check_update_config(config_path:str, **kwargs):
    '''对config做一些参数检查和更新操作'''

    config = dict()
    if config_path is not None:
        config.update(json.load(open(config_path, encoding='utf-8')))
    config.update(kwargs)
    if 'max_position' not in config:
        config['max_position'] = config.get('max_position_embeddings', 512)
    if 'dropout_rate' not in config:
        config['dropout_rate'] = config.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in config:
        config['segment_vocab_size'] = config.get('type_vocab_size', 2)

    # 获取_attn_implementation的配置项, 自动进行一些设置
    _attn_implementation = config.get('_attn_implementation', config.get('flash_attention'))  # 兼容老配置文件
    if _attn_implementation is None:
        config['_attn_implementation'] = 'eager'
    elif (_attn_implementation in {True, 'sdpa'}) and (not is_torch_sdpa_available()):
        log_warn_once('`F.scaled_dot_product_attention` only supported in torch 2.0')
        config['_attn_implementation'] = 'eager'
    elif (_attn_implementation == 'xformers') and (not is_xformers_available()):
        log_warn_once("Xformers is not installed correctly. use `pip install xformers`.")
        config['_attn_implementation'] = 'eager'
    elif (_attn_implementation == 'flash_attn_2') and (not is_flash_attn_available()):
        log_warn_once("flash_attn is not installed correctly. please visit https://github.com/Dao-AILab/flash-attention")
        config['_attn_implementation'] = 'eager'

    return DottableDict(config)