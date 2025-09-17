from .llm import CHAT_START_DOCSTRING, OPENAI_START_DOCSTRING
from .llm import LLM_MAPPING, ChatCli, ChatWebGradio, ChatWebStreamlit, ChatOpenaiApi, PretrainedTextContinuation
from .vlm import VLM_MAPPING, ChatVLCli, ChatVLWebGradio, ChatVLWebStreamlit, ChatVLOpenaiApi
from argparse import REMAINDER, ArgumentParser
from typing import List, Literal
import json
from bert4torch.snippets import get_config_path, log_info_once, add_start_docstrings


class Chat:
    """
    部署类似OpenAi的api server端

    ### 基础参数
    :param checkpoint_path: str, 模型所在的文件夹地址
    :param torch_dtype: bool, 精度, 'double', 'float', 'half', 'float16', 'bfloat16'
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quant_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'max_length':2048, 'default_rtype':'logits', 'use_states':True}
        - bos_token_id: int, 解码使用的起始token_id, 不同预训练模型设置可能不一样
        - eos_token_id: int/tuple/list, 解码使用的结束token_id, 不同预训练模型设置可能不一样, 默认给的-1(真实场景中不存在, 表示输出到max_length)
        - max_new_tokens: int, 最大解码长度
        - min_new_tokens: int, 最小解码长度, 默认为1
        - max_length: int, 最大文本长度
        - pad_token_id: int, pad_id, 在batch解码时候使用
        - padding_side: str, padding在前面还是后面, left或者right
        - device: str, 默认为'cpu'
        - n: int, random_sample时候表示生成的个数; beam_search时表示束宽
        - top_k: int, 这里的topk是指仅保留topk的值 (仅在top_k上进行概率采样)
        - top_p: float, 这里的topp是token的概率阈值设置(仅在头部top_p上进行概率采样)
        - temperature: float, 温度参数, 默认为1, 越小结果越确定, 越大结果越多样
        - repetition_penalty: float, 重复的惩罚系数, 越大结果越不重复
        - min_ends: int, 最小的end_id的个数
    :param create_model_at_startup: bool, 是否在启动的时候加载模型, 默认为True
    :param system: Optional[str]=None, 模型使用的system信息, 仅部分模型可用, 且openai api格式的不需要设置该参数

    ### 模式
    :param mode: 命令行, web, api服务模式, Literal['raw', 'cli', 'gradio', 'streamlit', 'openai']
    :param template: 使用的模板, 一般在bert4torch_config.json中无需单独设置, 可自行指定

    ### openai api参数
    :param model_name: str, 模型名称
    :param route_api: str, api的路由
    :param route_models: str, 模型列表的路由
    :param offload_when_nocall: str, 是否在一定时长内无调用就卸载模型，可以卸载到内存和disk两种
    :param offload_max_callapi_interval: int, 最长调用间隔
    :param offload_scheduler_interval: int, 定时任务的执行间隔
    :param api_keys: List[str], api keys的list

    ### Examples:
    ```python
    >>> from bert4torch.pipelines import Chat

    >>> checkpoint_path = "E:/data/pretrain_ckpt/glm/chatglm2-6b"
    >>> generation_config  = {'mode':'random_sample',
    ...                     'max_length':2048, 
    ...                     'default_rtype':'logits', 
    ...                     'use_states':True
    ...                     }
    >>> chat = Chat(checkpoint_path, generation_config=generation_config, mode='cli')
    >>> chat.run()
    ```
    """
    def __init__(self, 
                 # 基类使用
                 checkpoint_path:str, 
                 config_path:str=None,
                 torch_dtype:Literal['double', 'float', 'half', 'float16', 'bfloat16', None]=None, 
                 quantization_config:dict=None, 
                 generation_config:dict=None, 
                 create_model_at_startup:bool=True,
                 # cli参数
                 system:str=None,
                 # openapi参数
                 model_name:str='default', 
                 route_api:str='/chat/completions', 
                 route_models:str='/models', 
                 offload_max_callapi_interval:int=24*3600, 
                 offload_scheduler_interval:int=10*60, 
                 offload_when_nocall:Literal['cpu', 'disk', 'delete']=None, 
                 api_keys:List[str]=None,
                 # 模式
                 mode:Literal['raw', 'cli', 'gradio', 'streamlit', 'openai']='raw',
                 template: str=None,
                 **kwargs
                 ) -> None:
        pass

    def __new__(cls, *args, mode:Literal['raw', 'cli', 'gradio', 'streamlit', 'openai']='raw', **kwargs):
        # template指定使用的模板
        if kwargs.get('template') is not None:
            template = kwargs.pop('template')
        else:
            config_path = kwargs['config_path'] if kwargs.get('config_path') is not None else args[0]
            config = json.load(open(get_config_path(config_path, allow_none=True), encoding='utf-8'))
            template = config.get('template', config.get('model', config.get('model_type')))

        if template is None:
            raise ValueError('template/model/model_type not found in bert4torch_config.json')
        elif template in LLM_MAPPING:
            # 大语言模型
            ChatTemplate = LLM_MAPPING[template]
        elif template in VLM_MAPPING:
            # 多模态模型
            ChatTemplate = VLM_MAPPING[template]
        else:
            # 续写
            template = 'pretrained_text_continuation'
            ChatTemplate = PretrainedTextContinuation

        if template == 'pretrained_text_continuation':
            log_info_once('PretrainedTextContinuation is used, only can continue your text.')

        if mode == 'cli':
            @add_start_docstrings(CHAT_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVLCli if template in VLM_MAPPING else ChatCli): pass
        elif mode == 'gradio':
            @add_start_docstrings(CHAT_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVLWebGradio if template in VLM_MAPPING else ChatWebGradio): pass
        elif mode == 'streamlit':
            @add_start_docstrings(CHAT_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVLWebStreamlit if template in VLM_MAPPING else ChatWebStreamlit): pass
        elif mode == 'openai':
            @add_start_docstrings(OPENAI_START_DOCSTRING)
            class ChatDemo(ChatTemplate, ChatVLOpenaiApi if template in VLM_MAPPING else ChatOpenaiApi): pass
        elif mode == 'raw':
            ChatDemo = ChatTemplate
        else:
            raise ValueError(f'Unsupported mode={mode}')
        return ChatDemo(*args, **kwargs)


# ==========================================================================================
# =========================              命令行参数调用          ============================
# ==========================================================================================
def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="Bert4torch Pipelines LLM Server Launcher")

    parser.add_argument("--checkpoint_path", type=str, help="pretrained model name or path")
    parser.add_argument("--config_path", type=str, default=None, 
                        help="bert4torch_config.json file path or pretrained_model_name_or_path, if not set use `checkpoint_path` instead")
    parser.add_argument("--mode", type=str, choices=['cli', 'gradio', 'openai'], default='cli', 
                        help="deploy model in cli / gradio / openai mode")
    parser.add_argument("--torch_dtype", type=str, choices=['double', 'float', 'half', 'float16', 'bfloat16', None], default=None, 
                        help="modify model torch_dtype")
    
    # 命令行参数
    parser.add_argument("--system", type=str, default=None, help="cli args: model system/prompt/instrunctions")
    parser.add_argument("--functions", type=list, default=None, help="cli args: functions")

    # generation_config
    parser.add_argument("--top_k", type=int, default=None, help="generation_config: top_k")
    parser.add_argument("--top_p", type=float, default=None, help="generation_config: top_p")
    parser.add_argument("--temperature", type=float, default=None, help="generation_config: temperature")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="generation_config: repetition_penalty")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="generation_config: max_new_tokens")
    parser.add_argument("--max_length", type=int, default=None, help="generation_config: max_length")

    # quantization_config: 量化参数，显存不够时候可使用
    parser.add_argument("--quant_method", type=str, default=None, choices=['cpm_kernels', 'load_in_8bit', 'load_in_4bit'], 
                        help="quantization_config: quant_method")
    parser.add_argument("--quantization_config_others", type=dict, default=None, help="quantization_config: quantization_config_others")

    # openai参数
    parser.add_argument("--create_model_at_startup", type=bool, default=True, help="openai api args: whether create model at startup")
    parser.add_argument("--model_name", type=str, default='default', help="openai api args: model name")
    parser.add_argument("--route_api", type=str, default='/chat/completions', help="openai api args: `/chat/completions` route url")
    parser.add_argument("--route_models", type=str, default='/models', help="openai api args: `/models` route url")
    parser.add_argument("--api_keys", type=List[str], default=None, help="openai api args: authorized api keys list")
    parser.add_argument("--offload_when_nocall", type=str, choices=['cpu', 'disk', 'delete'], default=None, help="openai api args: ")
    parser.add_argument("--offload_max_callapi_interval", type=int, default=24*3600, help="openai api args: ")
    parser.add_argument("--offload_scheduler_interval", type=int, default=10*60, help="openai api args: ")
    
    # host和port
    parser.add_argument("--host", type=str, default='0.0.0.0', help="server host")
    parser.add_argument("--port", type=int, default=8000, help="server port")

    args = parser.parse_args()
    generation_config = {
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length
        }
    args.generation_config = {k: v for k, v in generation_config.items() if v is not None}

    if args.quant_method is not None:
        quantization_config = {"quant_method": args.quant_method}
        if args.quantization_config_others is not None and isinstance(args.quantization_config_others, dict):
            quantization_config.update(args.quantization_config_others)
        args.quantization_config = quantization_config
    return args


def run_llm_serve():
    '''命令行bert4torch serve直接部署模型'''
    args = get_args_parser()

    demo = Chat(args.checkpoint_path, 
                mode = args.mode,
                system = args.system,
                config_path = getattr(args, 'config_path', None),
                generation_config = args.generation_config,
                quantization_config = getattr(args, 'quantization_config', None),
                model_name=args.model_name,
                create_model_at_startup = args.create_model_at_startup,
                offload_when_nocall = args.offload_when_nocall,
                offload_max_callapi_interval = args.offload_max_callapi_interval,
                offload_scheduler_interval = args.offload_scheduler_interval
                )
    if args.mode == 'cli':
        demo.run(functions = getattr(args, 'functions', None))
    elif args.mode == 'gradio':
        demo.run(host=args.host, port=args.port)
    # elif args.mode == 'streamlit':
    #     demo.run()
    elif args.mode == 'openai':
        demo.run(host=args.host, port=args.port)
    else:
        raise ValueError(f'Args `mode`={args.mode} not supported')
