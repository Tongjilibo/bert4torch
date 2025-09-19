''' 模型
    v0.2.2版本前Trainer是在bert4torch内部实现的，之后单独为Trainer做了一个包torch4keras
    v0.2.5版本开始，对抗训练模块不在complile中使用，而是用callback方式实现
'''
import torch
from torch import nn
from bert4torch.layers import LayerNorm
from bert4torch.models.modeling_utils import load_state_dict_into_meta_model
from bert4torch.snippets import (
    JsonConfig, 
    log_warn, 
    find_tied_parameters, 
    print_trainable_parameters,
    get_parameter_device, 
    load_checkpoint, 
    save_checkpoint, 
    copytree, 
    log_info, 
    log_warn,
    log_warn_once,
    is_accelerate_available,
    DottableDict,
    has_meta_param
)
from torch4keras.model import BaseModel, add_trainer
import warnings
from typing import Union, Literal, Callable, List
from tqdm import tqdm
import gc
import copy
import re
import os


if is_accelerate_available():
    from accelerate import dispatch_model


class PreTrainedModel(nn.Module):
    """模型基类
    """
    def __init__(
            self,
            initializer_range:float=0.02,  # 权重初始化方差
            keep_tokens:List[int]=None,  # 要保留的词ID列表
            compound_tokens:List[int]=None,  # 扩展Embedding
            **kwargs
    ):
        super(PreTrainedModel, self).__init__()
        self.initializer_range = initializer_range
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.quantized = False
        self.add_trainer = kwargs.get('add_trainer', False)
        self.dtype = None
        self.config = None

    def tie_weights(self):
        pass
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpoint=True

    def enable_input_require_grads(self):
        """transformer移植来
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """transformer移植来
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()

    def get_kw(self, *args, **kwargs):
        '''把self.属性设置到kwargs中, 方便传参'''
        for arg in args:
            kwargs[arg] = getattr(self, arg)
        return kwargs

    def args_segmentate(self, inputs, **model_kwargs):
        '''解析输入，转成list，tuple类型'''
        # 传入[x1,x2]时，*inputs会解析成([x1,x2],)，此时需要取第一个元素
        if (len(inputs)==1) and isinstance(inputs[0], (tuple,list)):
            return inputs[0]
        return inputs

    def forward(self, *inputs, **model_kwargs):
        """定义模型的训练流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), attention_mask(若有), conditional_emb(若有), additional_embs(若有)]]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        # 允许model([token_ids, segment_ids]), model(token_ids, segment_ids)调用方式
        inputs = self.args_segmentate(inputs, **model_kwargs)
        # Embedding
        model_kwargs = self.apply_embeddings(*inputs, **model_kwargs)
        # Main
        model_kwargs = self.apply_main_layers(**model_kwargs)
        # Final
        outputs = self.apply_final_layers(**model_kwargs)

        if model_kwargs.get('use_states', False):
            return outputs, model_kwargs
        return outputs

    @torch.no_grad()
    def predict(self, *inputs, **model_kwargs):
        """定义模型的预测流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), attention_mask(若有), conditional_emb(若有), additional_embs(若有)]]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        if self.training:
            self.eval()
        return self.forward(*inputs, **model_kwargs)

    def init_model_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, (nn.Linear, nn.Embedding)) and (module.weight.requires_grad):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 固定的相对位置编码如Sinusoidal无需初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and (module.bias is not None) and module.bias.requires_grad:  # T5等模型使用的是rmsnorm
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and (module.bias is not None) and (module.bias.requires_grad):
            module.bias.data.zero_()

    def init_meta_weights(self, module):
        '''meta weights初始化, 主要是在量化里面用到
        '''
        if hasattr(module, 'weight') and module.weight.device == torch.device('meta'):
            module.to_empty(device='cpu')

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表"""
        return {}

    def load_variable(self, *args, **kwargs):
        raise NotImplementedError

    def load_embeddings(self, embeddings):
        """根据keep_tokens和compound_tokens对embedding进行修改"""
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                try:
                    ext_embeddings.append(torch.mean(embeddings[item], 0) * torch.ones_like(embeddings[item]))
                except IndexError:
                    ext_embeddings.append(torch.mean(embeddings, 0, keepdim=True))
                    warnings.warn(f'Initialize ext_embeddings from compound_tokens not in embedding index')
            embeddings = torch.cat([embeddings] + ext_embeddings, 0)

        return embeddings

    def load_trans_ckpt(self, checkpoint:str):
        """加载ckpt并转换
           1. 支持.safe_tensors + .bin
           2. 方便后续各个模型继承并做一些预处理, 如对qkv权重进行split
        """
        return load_checkpoint(checkpoint)

    def from_pretrained_single(
        self, 
        checkpoint:Union[str, os.PathLike]=None, 
        mapping:Union[dict,Callable]=None,
        device_map:dict=None, 
        torch_dtype=None, 
        verbose=1
    ):
        """加载预训练模型(单个权重文件)，根据mapping从checkpoint加载权重"""
        # 加载模型文件, 并可专业些转换
        ckpt_state_dict = self.load_trans_ckpt(checkpoint)
        
        state_dict_new = {}  # 用model_key作为key整理后的权重字典
        missing_keys = []  # 即model-ckpt, 当前加载中没有成功加载的权重ckpt_key名
        over_keys = set(ckpt_state_dict.keys())  # ckpt-model
        needed_keys = []  # 所需要的全部的ckpt_key名
        model_state_dict = self.state_dict()  # 模型的字典

        # 计算mapping
        mapping = mapping or self.variable_mapping()
        model_params = set([i[0] for i in self.named_parameters()])  # 可更新的变量
        if isinstance(mapping, dict):
            for layer_name in model_params:
                if layer_name in mapping:
                    continue
                if layer_name in ckpt_state_dict:
                    # 不在预设的mapping中, 但是ckpt和model中同时存在，则更新mapping
                    # 主要是为了在外部继承BERT后有其他layer，也能自动从checkpoint中加载进来
                    mapping.update({layer_name: layer_name})
                else:
                    # TODO: 不在预设的mapping中, 但是在model中
                    # 如果权重文件只有一个，则这些参数随机初始化的，其实应该算missing_keys
                    # 如果权重文件有多个，则这些参数可能在其shards中，这里添加到missing_keys，在外面统一清算
                    missing_keys.append(layer_name)
        elif isinstance(mapping, Callable):
            mapping = {mapping(k):k for k in ckpt_state_dict}
        else:
            raise TypeError(f'Args `mapping`={type(mapping)} not supported')

        # 加载parameter
        for model_key, ckpt_key in mapping.items():
            # 1. mapping和model不一致则忽略，如with_nsp=False时候在mapping中有但是model中没有
            if model_key not in model_state_dict:
                continue

            # 2. model中有，且ckpt中有，正常加载
            if ckpt_key in ckpt_state_dict:
                state_dict_new[model_key] = self.load_variable(ckpt_state_dict[ckpt_key], ckpt_key, model_key)
                # 去除已加载的Parameter，仅保留未能加载预训练权重的Parameter
                if ckpt_key in over_keys:
                    over_keys.remove(ckpt_key)
            
            # 3. model中有，但ckpt中没有，即ckpt中缺失部分参数
            else:
                missing_keys.append(ckpt_key)
            needed_keys.append(ckpt_key)
        
        over_keys = list(over_keys)
        del ckpt_state_dict
        gc.collect()
        
        self._print_mismatch_keys(missing_keys, over_keys, verbose)  # 打印mixmatch keys

        # 将ckpt的权重load到模型结构中
        if has_meta_param(self):
            load_state_dict_into_meta_model(self, state_dict_new, device_map=device_map, dtype=torch_dtype, 
                                            is_safetensors=checkpoint.endswith(".safetensors"))
        else:
            self.load_state_dict(state_dict_new, strict=False)
            
        del state_dict_new
        gc.collect()
        return missing_keys, over_keys, needed_keys

    def from_pretrained(
        self, 
        checkpoints:Union[str, os.PathLike, list], 
        mapping:Union[dict, Callable]=None, 
        device_map:dict=None, 
        torch_dtype=None, 
        verbose=1,
        **kwargs
    ):
        """加载预训练模型(单个/多个ckpt)"""
        self.dtype = torch_dtype
        
        # 单个权重文件
        if isinstance(checkpoints, str):
            self.from_pretrained_single(checkpoints, mapping=mapping, device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
        elif isinstance(checkpoints, (tuple, list)) and len(checkpoints)==1:
            self.from_pretrained_single(checkpoints[0], mapping=mapping, device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
        # 多个权重文件
        elif isinstance(checkpoints, (tuple, list)):
            all_needed_keys, all_missing_keys, all_over_keys = [], [], []
            tqdm_checkpoints = tqdm(checkpoints)
            for checkpoint in tqdm_checkpoints:
                tqdm_checkpoints.set_description(f'Loading {os.path.basename(checkpoint)}')
                missing_keys, over_keys, needed_keys = \
                    self.from_pretrained_single(checkpoint, mapping=mapping, device_map=device_map, torch_dtype=torch_dtype, verbose=0)
                all_needed_keys.extend(needed_keys)
                all_missing_keys.extend(missing_keys)
                all_over_keys.extend(over_keys)
                if checkpoint == checkpoints[-1]:
                    tqdm_checkpoints.set_description('Loading checkpoint shards')
                                             
            # 打印mixmatch keys
            all_missing_keys = set(all_missing_keys).difference(set(all_needed_keys))
            all_over_keys = set(all_over_keys).difference(set(all_needed_keys))
            self._print_mismatch_keys(all_missing_keys, all_over_keys, verbose)

        else:
            raise ValueError('Args `checkpoint_path` only support `str` or `list(str)` format')

        # Dispatch model with hooks on all devices if necessary
        if (device_map is not None) and is_accelerate_available():
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": kwargs.get('offload_folder'),
                "offload_index": kwargs.get('offload_index'),
                "offload_buffers": kwargs.get('offload_buffers', False),
                'skip_keys': 'past_key_values'
            }
            dispatch_model(self, **device_map_kwargs)

    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # if the module does not appear in _no_split_modules, we also check the children
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, PreTrainedModel):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)
    
    @staticmethod
    def _print_mismatch_keys(missing_keys, over_keys, verbose):
        """打印mismatch keys"""
        if verbose != 0:
            # model中有，但是ckpt中不存在
            if missing_keys:
                log_warn(f'`{missing_keys}` not found in pretrained checkpoints')
        if verbose > 1:
            # ckpt中存在，但是model中不存在
            if over_keys:
                log_warn(f'`{over_keys}` only exists in pretrained checkpoints but not in model parameters')

    def save_trans_ckpt(self):
        """对state_dict进行转换
        1. load_trans_ckpt的逆操作
        2. 方便后续各个模型继承并做一些预处理, 如合并qkv权重
        """
        return self.state_dict()

    def save_pretrained(self, save_path:str, weight_map:dict=None, mapping:Union[dict,Callable]=None, write_to_disk:bool=True, ignore_tied_parameters=False):
        '''按照预训练模型的key来保存模型, 可供transformers包加载
           1. 按照variable_mapping()逆向来保存权重
           2. 各个模型存在save_trans_ckpt()的也要执行, 如部分大模型需要把q,k,v合并为qkv

           :param save_path: str, 保存的文件/文件夹路径
           :param weight_map: dict, 部分大模型会有pytorch_model.bin.index.json文件, 对应其中的weight_map字段
                              可`from bert4torch.snippets import JsonConfig
                                 weight_map = JsonConfig(config_path).weight_map`来加载
           :param mapping: dict/func, 一般来说为None, 也允许用户自行指定映射关系（一般不需要）
           :param write_to_disk: bool, 是否写入硬盘，一般都是True, 该参数主要是为了在Trainer().save_pretrained
           :param ignore_tied_parameters: bool, 保存时候忽视tied_parameters
        '''
        state_dict = self.save_trans_ckpt()
        
        if ignore_tied_parameters:
            named_tied_parameters = find_tied_parameters(self)
            tied_parameters = [tied_parameter for _, tied_parameters in named_tied_parameters.items() \
                               for tied_parameter in tied_parameters]
            log_info(f'Remove tied parameters: {tied_parameters}')
            for tied_parameter in tied_parameters:
                    if tied_parameter in state_dict:
                        state_dict.pop(tied_parameter)
        
        mapping = mapping or self.variable_mapping()
        for k in list(state_dict.keys()):
            if isinstance(mapping, dict):
                state_dict[mapping.get(k, k)] = state_dict.pop(k)
            elif isinstance(mapping, Callable):
                state_dict[mapping(k)] = state_dict.pop(k)
        
        # 如果save_path是文件夹，则把对应的其他文件copy过去
        save_dir = None if re.search(r'\.[a-zA-z0-9]+$', save_path) else save_path

        # 把checkpoint_path所在目录下，除了权重文件的其他文件copy过去
        if write_to_disk and hasattr(self, 'checkpoint_path') and (self.checkpoint_path is not None) and save_dir:
            if isinstance(self.checkpoint_path, str):
                checkpoint_dir = os.path.dirname(self.checkpoint_path) if os.path.isfile(self.checkpoint_path) else self.checkpoint_path
            elif isinstance(self.checkpoint_path, (tuple, list)):
                checkpoint_dir = os.path.dirname(self.checkpoint_path[0]) if os.path.isfile(self.checkpoint_path[0]) else self.checkpoint_path[0]
            else:
                raise TypeError(f'`self.checkpoint_path` only support str,tuple,list')

            copytree(checkpoint_dir, save_dir, ignore_copy_files=[r'\.bin$', r'\.safetensors$'], dirs_exist_ok=True)  # 如果目录下文件存在也会强制覆盖

            # checkpoint shards对应的.index.json
            bin_index_json = [os.path.join(checkpoint_dir, i) for i in os.listdir(checkpoint_dir) if i.endswith('.index.json')]
            bin_index_json = bin_index_json[0] if bin_index_json else ''
            if (save_dir is not None) and os.path.exists(bin_index_json):
                weight_map = weight_map or JsonConfig(bin_index_json).get('weight_map')

        # 保存为单文件
        if weight_map is None:
            if write_to_disk:
                save_checkpoint(state_dict, os.path.join(save_dir, 'pytorch_model.bin') if save_dir else save_path)
            else:
                return state_dict
        
        # 保存为多个文件
        else:
            ckpt2param = dict()
            for param_name, save_file in weight_map.items():
                if save_file not in ckpt2param:
                    ckpt2param[save_file] = set([param_name])
                else:
                    ckpt2param[save_file].add(param_name)
            
            for save_file, param_names in ckpt2param.items():
                single_ckpt = {}
                for k in list(state_dict.keys()):
                    if k in param_names:
                        single_ckpt[k] = state_dict.pop(k)
                save_checkpoint(single_ckpt, os.path.join(save_dir or save_path, save_file))
        
    def apply_embeddings(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_main_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_final_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError
    
    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias"""
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）"""
        return self.position_bias

    def set_outputs(self, outputs):
        """设置output和oututs属性"""
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    def quantize(self, quant_method:Literal['cpm_kernels', 'load_in_8bit', 'load_in_4bit', 'gptq', 'awq'], **kwargs):
        '''量化
        1. 前量化: 模型权重已量化, 模型权重加载前即修改模型结构, 此时quantization_config在bert4torch_config.json里面
        2. 后量化: 模型权重未量化, 模型权重加载后再修改模型结构
            - build_transformer_model()中传入quantization_config
            - build_transformer_model()完成后, 再手动调用model.quantize()
            - pipelines.Chat()中传入quantization_config
        
        Examples:
        ```python
        >>> # 1. bitsandbytes的load_in_8bit量化
        >>> model = model.quantize(quant_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
        
        >>> # 2. bitsandbytes的load_in_4bit量化
        >>> from transformers import BitsAndBytesConfig
        >>> q_config = BitsAndBytesConfig(load_in_4bit=True,
        ...                             bnb_4bit_quant_type='nf4',
        ...                             bnb_4bit_use_double_quant=True,
        ...                             bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
        ...                             llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
        ...                             )
        >>> model = model.quantize(quant_method='load_in_4bit', quantization_config=q_config)
        
        >>> # 3. cpm_kernels量化
        >>> model = model.quantize(quant_method='cpm_kernels', quantization_bit=8)
        ```
        '''
        if self.quantized:
            print("Already quantized.")
            return self
        
        quantization_config = DottableDict(copy.deepcopy(kwargs))
        if 'model' in quantization_config:
            quantization_config.pop('model')
        
        from bert4torch.quantizers.auto import AUTO_QUANTIZER_MAPPING
        from bert4torch.quantizers.base import QuantizerBase
        quantization_config['quant_method'] = quant_method
        torch_dtype = quantization_config.pop('torch_dtype', None)
        device_map = quantization_config.pop('device_map', None)
        quantizer:QuantizerBase = AUTO_QUANTIZER_MAPPING[quant_method](quantization_config=quantization_config)
        quantizer.validate_environment(
            torch_dtype=torch_dtype,
            device_map=device_map,
            weights_only=True,
        )
        torch_dtype = quantizer.update_torch_dtype(torch_dtype)
        device_map = quantizer.update_device_map(device_map)
        quantization_config = quantizer.update_tp_plan(quantization_config)

        quantizer.preprocess_model(model=self, device_map=device_map, quantization_config=quantization_config)

        self.quantized = True
        self.quantizer = quantizer
        torch.cuda.empty_cache()
        return self

    def add_adapter(self, adapter_method='bottleneck', bottlenect_size=64):
        '''增加adapter层'''
        from bert4torch.layers import add_adapter
        self = add_adapter(self, adapter_method, bottlenect_size)
        self.print_trainable_parameters()
        return self
        
    def get_peft_model(self, peft_config, adapter_name="default"):
        '''hf的peft库：https://github.com/huggingface/peft
        peft的接口LoraModel接口有变，这里使用v0.0.3
        '''
        import peft
        self.peft_config = {adapter_name: peft_config}
        if isinstance(peft_config, peft.LoraConfig):
            model = peft.LoraModel(self, self.peft_config, adapter_name)
        elif isinstance(peft_config, peft.AdaLoraConfig):
            model = peft.AdaLoraModel(self, self.peft_config, adapter_name)
        else:
            raise ValueError(f'{type(peft_config)} has not been supported')
        
        # 返回的model无法使用torch4keras的trainer
        self = add_trainer(model) if self.add_trainer else model
        self.print_trainable_parameters()
        return self

    def print_trainable_parameters(self):
        """打印可训练的参数量"""
        print_trainable_parameters(self)
    
    @property
    def device(self) -> torch.device:
        """获取model所在的device"""
        return get_parameter_device(self)


def extend_with_base_model(InputModel):
    """添加torch4keras的BaseModel, 可以使用.compile, .fit等Trainer的功能"""
    class BertBaseModel(InputModel, PreTrainedModel, BaseModel):
        pass
    return BertBaseModel


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）"""
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask"""
        token_ids = inputs[0]
        seq_len = token_ids.shape[1]
        attention_bias = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device), diagonal=0)
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias


def extend_with_language_model(InputModel):
    """添加下三角的Attention Mask（语言模型用）"""
    class LanguageModel(LM_Mask, InputModel):
        """带下三角Attention Mask的派生模型"""
        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(LanguageModel, self).__init__(*args, **kwargs)

    return LanguageModel


class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）；
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask"""
        segment_ids = inputs[1]
        attention_bias = torch.cumsum(segment_ids, dim=1)
        attention_bias = (attention_bias.unsqueeze(1)) <= (attention_bias.unsqueeze(2))
        self.attention_bias = attention_bias.unsqueeze(1).long()

        return self.attention_bias


def extend_with_unified_language_model(InputModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）"""
    class UnifiedLanguageModel(UniLM_Mask, InputModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)

    return UnifiedLanguageModel