''' 模型
    v0.2.2版本前Trainer是在bert4torch内部实现的，之后单独为Trainer做了一个包torch4keras
    v0.2.5版本开始，对抗训练模块不在complile中使用，而是用callback方式实现
'''
import torch
from torch import nn
from bert4torch.layers import LayerNorm
from bert4torch.snippets import torch_div, log_warn, load_state_dict_into_meta_model
from bert4torch.snippets import take_along_dim, get_parameter_device
import warnings
from torch4keras.model import *
from tqdm import tqdm
import gc
import copy


class BERT_BASE(nn.Module):
    """模型基类
    """
    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate=None,  # Dropout比例
            attention_probs_dropout_prob=None,  # Attention矩阵的Dropout比例
            embedding_size=None,  # 指定embedding_size, 不指定则使用config文件的参数
            attention_head_size=None,  # Attention中V的head_size
            attention_key_size=None,  # Attention中Q,K的head_size
            initializer_range=0.02,  # 权重初始化方差
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            keep_hidden_layers=None, # 保留的hidden_layer层的id
            hierarchical_position=None,  # 是否层次分解位置编码
            gradient_checkpoint=False, # 是否使用gradient_checkpoint
            output_all_encoded_layers=False, # 是否返回所有layer的hidden_states
            tie_emb_prj_weight=False,  # 是否绑定embedding和lm_head的权重
            return_dict=False,  # 是否返回的格式是dict
            **kwargs
    ):
        super(BERT_BASE, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)
        self.hierarchical_position = hierarchical_position
        self.gradient_checkpoint = gradient_checkpoint
        self.quantized = False
        self.output_all_encoded_layers = output_all_encoded_layers
        self.add_trainer = kwargs['add_trainer']
        self.tie_emb_prj_weight = tie_emb_prj_weight
        self.return_dict = return_dict

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
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
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
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
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

    def load_variable(self):
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

    def load_pos_embeddings(self, embeddings):
        """根据hierarchical_position对pos_embedding进行修改"""
        if self.hierarchical_position is not None:
            alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]
            # 为兼容低版本pytorch没有take_along_dim
            embeddings_x = take_along_dim(embeddings,  torch_div(position_index, embeddings.size(0), rounding_mode='trunc'), dim=0)  # 兼容老版本
            # embeddings_x = take_along_dim(embeddings,  torch.div(position_index, embeddings.size(0), rounding_mode='trunc'), dim=0)
            embeddings_y = take_along_dim(embeddings, position_index % embeddings.size(0), dim=0)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y

        return embeddings

    def load_trans_ckpt(self, checkpoint):
        """加载ckpt, 方便后续继承并做一些预处理"""
        if isinstance(checkpoint, str):
            return torch.load(checkpoint, map_location='cpu')
        raise ValueError('Args `checkpoint_path` only support `str` format')

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None, skip_init=False, device_map=None, 
                                             torch_dtype=None, verbose=1):
        """根据mapping从checkpoint加载权重"""
        # 加载模型文件, 并可专业些转换
        ckpt_state_dict = self.load_trans_ckpt(checkpoint)
        
        # 计算mapping
        mapping = mapping or self.variable_mapping()
        model_params = set([i[0] for i in self.named_parameters()])  # 可更新的变量
        # 如果ckpt和model中同时存在，且不在预设的mapping中，则更新mapping
        # 主要是为了在外部继承BERT后有其他layer，也能自动从checkpoint中加载进来
        for layer_name in model_params:
            if (layer_name in ckpt_state_dict) and (layer_name not in mapping):
                mapping.update({layer_name: layer_name})

        state_dict_new = {}  # 用new_key作为key整理后的权重字典
        missing_keys = []  # 即model-ckpt, 当前加载中没有成功加载的权重old_keys名
        over_keys = set(ckpt_state_dict.keys())  # ckpt-model
        needed_keys = []  # 所需要的全部的old_keys名
        model_state_dict = self.state_dict()  # 模型的字典
        for new_key, old_key in mapping.items():
            # 1. mapping和model不一致则忽略，如with_nsp=False时候在mapping中有但是model中没有
            if new_key not in model_state_dict:
                continue

            # 2. model中有，且ckpt中有，正常加载
            if old_key in ckpt_state_dict:
                state_dict_new[new_key] = self.load_variable(ckpt_state_dict, old_key)
                # 去除已加载的Parameter，仅保留未能加载预训练权重的Parameter
                if old_key in over_keys:
                    over_keys.remove(old_key)
            
            # 3. model中有，但ckpt中没有，即ckpt中缺失部分参数
            else:
                missing_keys.append(old_key)
            needed_keys.append(old_key)
        
        over_keys = list(over_keys)
        del ckpt_state_dict
        gc.collect()
        
        self._print_mismatch_keys(missing_keys, over_keys, verbose)  # 打印mixmatch keys

        # 将ckpt的权重load到模型结构中
        if not skip_init:
            self.load_state_dict(state_dict_new, strict=False)
        else:
            load_state_dict_into_meta_model(self, state_dict_new, device_map=device_map, torch_dtype=torch_dtype)
            
        del state_dict_new
        gc.collect()
        return missing_keys, over_keys, needed_keys

    @staticmethod
    def _print_mismatch_keys(missing_keys, over_keys, verbose):
        """打印mismatch keys"""
        if verbose != 0:
            for key in missing_keys:  # model中有，但是ckpt中不存在
                log_warn(f'`{key}` not found in pretrained checkpoints')
        if verbose > 1:
            for key in over_keys:  # ckpt中存在，但是model中不存在
                log_warn(f'`{key}` only exists in pretrained checkpoints but not in model parameters')

    def load_weights_from_pytorch_checkpoints(self, checkpoints, mapping=None, skip_init=False, device_map=None, 
                                              torch_dtype=None, verbose=1):
        """逐个ckpt加载"""
        if isinstance(checkpoints, str):
            self.load_weights_from_pytorch_checkpoint(checkpoints, mapping=mapping, skip_init=skip_init, 
                                                      device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
        elif isinstance(checkpoints, (tuple, list)):
            all_missing_keys, all_over_keys = [], []
            for checkpoint in tqdm(checkpoints, desc='Loading checkpoint shards'):
                missing_keys, over_keys, needed_keys = \
                    self.load_weights_from_pytorch_checkpoint(checkpoint, mapping=mapping, skip_init=skip_init, 
                                                              device_map=device_map, torch_dtype=torch_dtype, verbose=0)
                all_missing_keys.extend(missing_keys)
                all_over_keys.extend(over_keys)

            # 打印mixmatch keys
            all_missing_keys = set(all_missing_keys).difference(set(needed_keys))
            all_over_keys = set(all_over_keys).difference(set(needed_keys))
            self._print_mismatch_keys(all_missing_keys, all_over_keys, verbose)

        else:
            raise ValueError('Args `checkpoint_path` only support `str` or `list(str)` format')

    def apply_embeddings(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_main_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_final_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError
    
    def apply_on_layer_begin(self, l_i, **model_kwargs):
        '''新增对layer block输入进行操作的函数'''
        if ('past_key_values' not in model_kwargs) or (model_kwargs.get('past_key_values') is None):
            model_kwargs['past_key_value'] = None
        else:
            model_kwargs['past_key_value'] = model_kwargs['past_key_values'][l_i]

        if ('cross_past_key_values' not in model_kwargs) or (model_kwargs.get('cross_past_key_values') is None):
            model_kwargs['cross_past_key_value'] = None
        else:
            model_kwargs['cross_past_key_value'] = model_kwargs['cross_past_key_values'][l_i]
        return model_kwargs
    
    def apply_on_layer_end(self, l_i, **model_kwargs):
        '''新增对layer block输出进行操作的函数, 目前仅在MixUp中使用'''
        if model_kwargs.get('past_key_value', None) is not None:
            if ('past_key_values' not in model_kwargs) or (model_kwargs.get('past_key_values') is None):
                model_kwargs['past_key_values'] = [None]*self.num_hidden_layers
            model_kwargs['past_key_values'][l_i] = model_kwargs['past_key_value']
        if model_kwargs.get('cross_past_key_value', None) is not None:
            if ('cross_past_key_values' not in model_kwargs) or (model_kwargs.get('cross_past_key_values') is None):
                model_kwargs['cross_past_key_values'] = [None]*self.num_hidden_layers
            model_kwargs['cross_past_key_values'][l_i] = model_kwargs['cross_past_key_value']
        return model_kwargs

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

    def quantize(self, quantization_method, **kwargs):
        '''量化'''
        if self.quantized:
            print("Already quantized.")
            return self
        
        new_kwargs = copy.deepcopy(kwargs)
        if 'model' in new_kwargs:
            new_kwargs.pop('model')

        # chatglm的量化方式
        if quantization_method == 'cpm_kernels':
            from bert4torch.quantization import quantize_cpm_kernels
            self = quantize_cpm_kernels(self, **new_kwargs)
        # load_in_8bit, load_in_4bit
        elif quantization_method in {'load_in_8bit', 'load_in_4bit'}:
            from bert4torch.quantization import quantize_load_in_kbit
            load_in_8bit = True if quantization_method == 'load_in_8bit' else False
            load_in_4bit = True if quantization_method == 'load_in_4bit' else False
            self = quantize_load_in_kbit(self, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **new_kwargs)
        else:
            raise ValueError('Please check args `quantization_method`')

        self.quantized = True
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
    
    
def extend_with_base_model(InputModel):
    """添加torch4keras的BaseModel"""
    class BertBaseModel(InputModel, BERT_BASE, BaseModel):
        pass
    return BertBaseModel