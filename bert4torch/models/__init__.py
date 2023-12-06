from torch4keras.model import *
from bert4torch.snippets import set_default_torch_dtype
from bert4torch.models.albert import *
from bert4torch.models.bart import *
from bert4torch.models.base import *
from bert4torch.models.bert import *
from bert4torch.models.deberta import *
from bert4torch.models.electra import *
from bert4torch.models.erinie import *
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


def build_transformer_model(config_path=None, checkpoint_path=None, model=None, application='encoder', add_trainer=False, verbose=1, **kwargs):
    """根据配置文件构建模型, 可选加载checkpoint权重

    :param config_path: str, 模型的config文件地址
    :param checkpoint_path: str/list[str], 模型文件地址, 默认值None表示不加载预训练模型
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
    :param skip_init: bool, 是否初始化, 默认为False
    :param low_cpu_mem_usage: bool, 是否初始化, 默认为False, 同skip_init, 仅需要设置一个即可
    :param device_map: None/str/dict, 为不同Module指定不同的device, 默认为None表示加载到cpu中, 不同于transformer自动分配, 这里需手动指定dict
    :param torch_dtype: 指定权重的dtype
    :param flash_attention: bool, 是否使用flash_attention, 即torch2的scaled_dot_product_attention(), 默认为False
    :param use_logn_attn: bool, 在attention模块中是否使用logn_attn
    :param multi_query_group_num: int, 使用MQA的头数
    :param ntk_alpha: float, rope外推使用ntk方法时的alhpa参数

    :return: A pytorch model instance
    """
    configs = DottableDict()
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    device_map = configs.pop('device_map', None)
    skip_init = configs.get('skip_init', False) or configs.get('low_cpu_mem_usage', False)
    skip_init = True if device_map is not None else skip_init  # 指定了device_map, 就必须skip_init

    torch_dtype = configs.pop('torch_dtype', None)
    configs['add_trainer'] = add_trainer

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
        'falcon': Falcon
    }

    model = model or configs.get('model', 'bert')
    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            configs['version'] = model
    elif isinstance(model, type) and issubclass(model, BERT_BASE): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('Args `model` type should be string or BERT_BASE')

    # 使用 lm/unilm
    application = application.lower()
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
    if skip_init:
        from accelerate import init_empty_weights
        with init_empty_weights():
            transformer = MODEL(**configs)
    else:
        transformer = MODEL(**configs)
        transformer.apply(transformer.init_model_weights)  # 初始化权重

    # 预训练模型是否已量化, 加载量化后的权重使用, 如果是加载原权重再自行量化这里不需要设置
    if configs.get('quantization_method') is not None:
        if skip_init:  # 把meta权重to_empty(device='cpu'), 执行后就不是meta了
            transformer.apply(transformer.init_meta_weights)
        transformer = transformer.half().quantize(**configs)
        skip_init = False

    # 恢复默认权重类型
    if dtype_orig is not None:
        torch.set_default_dtype(dtype_orig)
    
    # 权重加载
    checkpoint_path = checkpoint_path or configs.get('checkpoint_path')
    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoints(checkpoint_path, mapping=configs.get('mapping'), skip_init=skip_init, 
                                                          device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
    
    # 权重tie, 若skip_init则模型结构中的tie_weights会失效, 这里重新tie_weights一下
    transformer.tie_weights()
    transformer.configs = transformer.config = configs
    return transformer
