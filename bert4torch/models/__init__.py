from typing import Union
from torch4keras.model import *
from bert4torch.snippets import set_default_torch_dtype, get_checkpoint_path, get_config_path, is_accelerate_available
from bert4torch.models.albert import *
from bert4torch.models.bart import *
from bert4torch.models.base import *
from bert4torch.models.bert import *
from bert4torch.models.deberta import *
from bert4torch.models.electra import *
from bert4torch.models.ernie import *
from bert4torch.models.gau_alpha import *
from bert4torch.models.glm import *
from bert4torch.models.gpt import *
from bert4torch.models.llama import *
from bert4torch.models.nezha import *
from bert4torch.models.roformer import *
from bert4torch.models.t5 import *
from bert4torch.models.transformer import *
from bert4torch.models.transformer_xl import *
from bert4torch.models.uie import *
from bert4torch.models.xlnet import *
from bert4torch.models.bloom import *
from bert4torch.models.qwen import *
from bert4torch.models.internlm import *
from bert4torch.models.falcon import *
from bert4torch.models.deepseek import *
from typing import Union, Literal


def build_transformer_model(config_path:Union[str, os.PathLike]=None, checkpoint_path:Union[str, os.PathLike, list]=None, model:Union[str, BERT_BASE]=None, 
                            application:Literal['encoder', 'lm', 'unilm']=None, 
                            add_trainer:bool=False, verbose:int=1, **kwargs) -> Union[BERT_BASE, BERT, Transformer, Trainer]:
    """根据配置文件构建模型, 可选加载checkpoint权重

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
    :param multi_query_group_num: int, 使用MQA的头数
    :param ntk_alpha: float, rope外推使用ntk方法时的alhpa参数

    :return: A pytorch model instance

    Example
    ----------------------
    支持几种加载方式
    >>> 1. 仅指定config_path: 从头初始化模型结构, 不加载预训练模型
    >>>    model = build_transformer_model('./model/bert4torch_config.json')

    >>> 2. 仅指定checkpoint_path: 
    >>>     2.1 文件夹路径: 自动寻找路径下的*.bin/*.safetensors权重文件 + bert4torch_config.json/config.json文件
    >>>         model = build_transformer_model(checkpoint_path='./model')
    >>>     2.2 文件路径/列表: 文件路径即权重路径/列表, config会从同级目录下寻找
    >>>         model = build_transformer_model(checkpoint_path='./pytorch_model.bin')
    >>>     2.3 model_name: hf上预训练权重名称, 会自动下载hf权重以及bert4torch_config.json文件
    >>>         model = build_transformer_model(checkpoint_path='bert-base-chinese')

    >>> 3. 同时指定config_path和checkpoint_path(本地路径名或model_name排列组合): 
    >>>     config_path = './model/bert4torch_config.json'  # 或'bert-base-chinese'
    >>>     checkpoint_path = './model/pytorch_model.bin'  # 或'bert-base-chinese'
    >>>     model = build_transformer_model(config_path, checkpoint_path)
    """
    # 校验checkpoint_path, config_path
    config_path = get_config_path(config_path if config_path is not None else checkpoint_path, **kwargs)
    checkpoint_path = get_checkpoint_path(checkpoint_path, **kwargs)
    if (config_path is None) and (checkpoint_path is not None):
        # 没有找到bert4torch_config.json，则从local的checkpoint_path去找
        config_path = get_config_path(checkpoint_path, **kwargs)

    config = DottableDict()
    if config_path is not None:
        config.update(json.load(open(config_path)))
    config.update(kwargs)
    if 'max_position' not in config:
        config['max_position'] = config.get('max_position_embeddings', 512)
    if 'dropout_rate' not in config:
        config['dropout_rate'] = config.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in config:
        config['segment_vocab_size'] = config.get('type_vocab_size', 2)
    device_map = config.pop('device_map', None)
    skip_init = config.get('skip_init', False) or config.get('low_cpu_mem_usage', False)
    skip_init = True if device_map is not None else skip_init  # 指定了device_map, 就必须skip_init
    torch_dtype = config.pop('torch_dtype', None)
    config['add_trainer'] = add_trainer
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
        'ernie': ERNIE,
        'deberta_v2': DebertaV2,
        'uie': UIE,
        'encoder': Encoder,
        'decoder': Decoder,
        'transformer': Transformer,
        'bart': BART,
        'gpt': GPT,
        'gpt2': GPT2,
        'gpt2_ml': GPT2_ML,
        'llama': LLaMA,
        'baichuan': Baichuan,
        'glm': GLM,
        'chatglm': GLM,
        'glm2': GLM2,
        'chatglm2': GLM2,
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
        'internlm': InternLM,
        'falcon': Falcon,
        'deepseek': DeepSeek
    }

    model = model or config.get('model', 'bert')
    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            config['version'] = model
    elif isinstance(model, type) and issubclass(model, BERT_BASE): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('Args `model` type should be string or BERT_BASE')

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
    dtype_orig = None
    if torch_dtype is not None:
        torch_dtype, dtype_orig = set_default_torch_dtype(torch_dtype, model)

    # 生成网络结构
    if skip_init and (checkpoint_path is not None):
        if is_accelerate_available():
            from accelerate import init_empty_weights
            with init_empty_weights():
                transformer = MODEL(**config)
        else:
            skip_init = False  # 若accelerate包不存在则先初始化模型参数
            log_warn('Package `accelerate` not available, use `pip install accelerate`')
    if not skip_init:
        transformer = MODEL(**config)
        transformer.apply(transformer.init_model_weights)  # 初始化权重

    # 预训练模型是否已量化, 加载量化后的权重使用, 如果是加载原权重再自行量化这里不需要设置
    if config.get('quantization_method') is not None:
        if skip_init:  # 把meta权重to_empty(device='cpu'), 执行后就不是meta了
            transformer.apply(transformer.init_meta_weights)
        transformer = transformer.half().quantize(**config)
        skip_init = False

    # 恢复默认权重类型
    if dtype_orig is not None:
        torch.set_default_dtype(dtype_orig)
    
    # 权重加载
    transformer.checkpoint_path = checkpoint_path
    if checkpoint_path is not None:
        transformer.from_pretrained(checkpoint_path, mapping=config.get('mapping'), skip_init=skip_init, 
                                    device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
    
    # meta device则报错
    meta_names = []
    for name_, para_ in transformer.named_parameters():
        if str(para_.device) == 'meta':
            meta_names.append(name_)
    if len(meta_names) > 0:
        log_error(f'Meta device not allowed: {meta_names}')
    
    # 权重tie, 若skip_init则模型结构中的tie_weights会失效, 这里重新tie_weights一下
    transformer.tie_weights()
    transformer.configs = transformer.config = config
    return transformer
