''' 模型
    v0.2.2版本前Trainer是在bert4torch内部实现的，之后单独为Trainer做了一个包torch4keras
    v0.2.5版本开始，对抗训练模块不在complile中使用，而是用callback方式实现
'''
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
import json
import re
from bert4torch.layers import LayerNorm, BertEmbeddings, BertLayer, BlockIdentity, T5Layer, GatedAttentionUnit, XlnetLayer
from bert4torch.layers import AdaptiveEmbedding, XlnetPositionsEncoding, ConvLayer
from bert4torch.snippets import insert_arguments, delete_arguments, print_trainable_parameters, torch_div, colorful
from bert4torch.snippets import take_along_dim, create_position_ids_start_at_padding, DottableDict, get_parameter_device
from bert4torch.activations import get_activation
import warnings
from torch4keras.model import *
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from tqdm import tqdm


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
        self.skip_init = kwargs['skip_init']
        self.add_trainer = kwargs['add_trainer']

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
        if self.skip_init == 'meta':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.to_empty(device='cpu')
        elif isinstance(module, (nn.Linear, nn.Embedding)) and (module.weight.requires_grad):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 固定的相对位置编码如Sinusoidal无需初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and module.bias.requires_grad:  # T5等模型使用的是rmsnorm
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and (module.bias is not None) and (module.bias.requires_grad):
            module.bias.data.zero_()

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

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None, verbose=1):
        """根据mapping从checkpoint加载权重"""
        # 加载模型文件
        if isinstance(checkpoint, str):
            file_state_dict = torch.load(checkpoint, map_location='cpu')
        else:
            raise ValueError('Args `checkpoint_path` only support `str` format')
        
        mapping = mapping or self.variable_mapping()
        model_params = set([i[0] for i in self.named_parameters()])  # 可更新的变量
        
        # 如果ckpt和model中同时存在，且不在预设的mapping中，则更新mapping
        # 主要是如为了在外部继承BERT后有其他layer，也能自动从checkpoint中加载进来
        for layer_name in model_params:
            if (layer_name in file_state_dict) and (layer_name not in mapping):
                mapping.update({layer_name: layer_name})

        state_dict_new = {}
        missing_keys, needed_keys = [], []  # 都是old_keys的名字
        for new_key, old_key in mapping.items():
            # mapping和model不一致，忽略，如with_nsp=False时候在mapping中有但是model中没有
            if new_key not in self.state_dict():
                continue
            # model中有，且ckpt中有，正常加载
            elif old_key in file_state_dict:
                state_dict_new[new_key] = self.load_variable(file_state_dict, old_key)
            # model中有，但ckpt中没有，ckpt中缺失部分参数
            elif old_key not in file_state_dict:
                missing_keys.append(old_key)
            # 保留未能加载预训练权重的Parameter
            if new_key in model_params:
                model_params.remove(new_key)
            needed_keys.append(old_key)
        del file_state_dict

        # mismatch keys的处理
        if verbose != 0:
            for key in missing_keys:
                print(colorful(f'[WARNING] {key} not found in pretrain models'))
            for key in model_params:
                print(colorful(f'[WARNING] Parameter {key} not loaded from pretrain models'))

        # 将ckpt的权重load到模型结构中
        self.load_state_dict(state_dict_new, strict=False)
        del state_dict_new
        return missing_keys, needed_keys

    def load_weights_from_pytorch_checkpoints(self, checkpoints, mapping=None, verbose=1):
        """逐个ckpt加载"""
        if isinstance(checkpoints, str):
            self.load_weights_from_pytorch_checkpoint(checkpoints, mapping=mapping, verbose=verbose)
        elif isinstance(checkpoints, (tuple, list)):
            all_missing_keys = []
            for checkpoint in tqdm(checkpoints, desc='Loading checkpoint shards'):
                missing_keys, needed_keys = self.load_weights_from_pytorch_checkpoint(checkpoint, mapping=mapping, verbose=0)
                all_missing_keys.extend(missing_keys)
            all_missing_set = set(all_missing_keys).difference(set(needed_keys))
            if verbose != 0:
                for key in all_missing_set:
                    print(colorful(f'[WARNING] {key} not found in pretrain models'))
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
        model_kwargs['past_key_value'] = model_kwargs.get('past_key_values', [None]*self.num_hidden_layers)[l_i]
        model_kwargs['cross_past_key_value'] = model_kwargs.get('cross_past_key_values', [None]*self.num_hidden_layers)[l_i]
        return model_kwargs
    
    def apply_on_layer_end(self, l_i, **model_kwargs):
        '''新增对layer block输出进行操作的函数, 目前仅在MixUp中使用'''
        if model_kwargs.get('past_key_value', None) is not None:
            if 'past_key_values' not in model_kwargs:
                model_kwargs['past_key_values'] = [None]*self.num_hidden_layers
            model_kwargs['past_key_values'][l_i] = model_kwargs['past_key_value']
        if model_kwargs.get('cross_past_key_value', None) is not None:
            if 'cross_past_key_values' not in model_kwargs:
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

    def quantize(self, bits: int, use_quantization_cache=True, empty_init=False, target_modules=None, **kwargs):
        '''量化'''
        if bits == 0:
            return

        from .quantization import quantize

        if self.quantized:
            print("Already quantized.")
            return self

        self.quantized = True
        self.quantization_bit = bits
        self = quantize(self, bits, empty_init=empty_init, use_quantization_cache=use_quantization_cache, target_modules=target_modules, **kwargs)
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
        
        # 返回的model无法使用torch4keras的trainer
        self =  add_trainer(model) if self.add_trainer else model
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


class BERT(BERT_BASE):
    """构建BERT模型
    """
    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            custom_position_ids=False,  # 是否自行传入位置id, True表示传入，False表示不传入，'start_at_padding'表示从padding_idx+1开始
            custom_attention_mask=False, # 是否自行传入attention_mask
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            conditional_size=None,  # conditional layer_norm
            additional_embs=False, # addtional_embeddng, 是否有额外的embedding, 比如加入词性，音调，word粒度的自定义embedding
            is_dropout=False,
            pad_token_id=0,  # 默认0是padding ids, 但是注意google的mt5padding不是0
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.pad_token_id = pad_token_id
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.additional_embs = additional_embs
        self.conditional_size = conditional_size
        self.embeddings = BertEmbeddings(**self.get_kw('vocab_size', 'embedding_size', 'hidden_size', 'max_position', 'segment_vocab_size', 
                                                       'shared_segment_embeddings', 'dropout_rate', 'conditional_size', **kwargs))
        layer = BertLayer(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                        'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'max_position', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            if self.with_nsp:
                # Next Sentence Prediction部分
                # nsp的输入为pooled_output, 所以with_pool为True是使用nsp的前提条件
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None

        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.embedding_size)  # 允许hidden_size和embedding_size不同
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.embedding_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
            if kwargs.get('tie_emb_prj_weight') is True:
                self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
        # 下述继承于BERT的有声明新的参数，在这里初始化不能统一初始化到

    def apply_embeddings(self, *inputs, **model_kwargs):
        """BERT的embedding是token、position、segment三者embedding之和

        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有), additional_input(若有)]
        :return: List[torch.Tensor], [hidden_states, attention_mask, conditional_emb, ...]
        """
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'

        # ========================= token_ids =========================
        if model_kwargs.get('token_ids') is not None:
            token_ids = model_kwargs['token_ids']
        else:
            token_ids = inputs[0]
            index_ = 1
        
        # ========================= segment_ids =========================
        if model_kwargs.get('segment_ids') is not None:
            segment_ids = model_kwargs['segment_ids']
        elif self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None

        # ========================= position_ids =========================
        # [btz, seq_len]
        if model_kwargs.get('position_ids') is not None:
            position_ids = model_kwargs['position_ids']
        elif self.custom_position_ids is True:  # 自定义position_ids
            position_ids = inputs[index_]
            index_ += 1
        elif self.custom_position_ids == 'start_at_padding':
            # 从padding位置开始
            position_ids = create_position_ids_start_at_padding(token_ids, self.pad_token_id)
        else:
            position_ids = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device).unsqueeze(0)
        model_kwargs['position_ids'] = position_ids

        # ========================= attention_mask =========================
        # 这里attention_mask表示传入[btz, seq_len], 而后续的attention_mask其实是extended_attention_mask[btz, 1, 1/q_len, seq_len]
        if model_kwargs.get('attention_mask') is not None:
            # attention_mask是根据token_ids生成的，因此外部需要重置下，目前是带cache解码时候使用
            attention_mask = model_kwargs['attention_mask']
        elif self.custom_attention_mask:
            attention_mask = inputs[index_].long()
            index_ += 1
        elif (not token_ids.requires_grad) and (token_ids.dtype in {torch.long, torch.int}): # 正常的token_ids
            attention_mask = (token_ids != self.pad_token_id).long()  # 默认0为mask_value
            if self.pad_token_id < 0:
                token_ids = token_ids * attention_mask
        else:  # 自定义word_embedding，目前仅有VAT中使用
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask  # 缓存上次用的attention_mask
        model_kwargs['input_attention_mask'] = attention_mask
        # 根据token_ids创建一个3D的attention mask矩阵，尺寸为[batch_size, 1, 1, to_seq_length]，
        # 目的是为了适配多头注意力机制，从而能广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        self.compute_attention_bias([token_ids, segment_ids])  # 根据lm或者unilm需要对mask做调整
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias  # 不可访问padding
            # attention_mask = self.attention_bias  # 可以访问padding
        # pytorch >= 1.5时候会导致StopIteration错误
        # https://github.com/huggingface/transformers/issues/3936
        # https://github.com/huggingface/transformers/issues/4189
        # https://github.com/huggingface/transformers/issues/3936
        try:
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # 兼容fp16
        except StopIteration:
            attention_mask = attention_mask.to(dtype=torch.float32)
        
        # ========================= conditional layer_norm =========================
        if model_kwargs.get('conditional_emb') is not None:
            conditional_emb = model_kwargs['layer_norm_ids']
        elif self.conditional_size is not None:
            conditional_emb = inputs[index_]
            index_ += 1
        else:
            conditional_emb = None

        # ========================= addtional_embeddng =========================
        # 比如加入词性，音调，word粒度的自定义embedding
        if model_kwargs.get('additional_embs') is not None:
            additional_embs = model_kwargs['additional_embs']
        elif self.additional_embs is True:
            additional_embs = inputs[index_]
            index_ += 1
        else:
            additional_embs = None
        additional_embs = [additional_embs] if isinstance(additional_embs, torch.Tensor) else additional_embs
        assert (additional_embs is None) or isinstance(additional_embs, (tuple, list))

        # 进入embedding层
        hidden_states = self.embeddings(token_ids, segment_ids, position_ids, conditional_emb, additional_embs)
        model_kwargs.update({'hidden_states': hidden_states, 'attention_mask':attention_mask, 'conditional_emb': conditional_emb})
        
        # 解析encoder_hidden_state, encoder_attention_mask
        if len(inputs[index_:]) >=2:
            model_kwargs['encoder_hidden_states'], model_kwargs['encoder_attention_mask'] = inputs[index_], inputs[index_+1]
        return model_kwargs

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        
        :param inputs: List[torch.Tensor], 默认顺序为[hidden_states, attention_mask, conditional_emb]
        :return: List[torch.Tensor], 默认顺序为[encoded_layers, conditional_emb]
        """
        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.encoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = grad_checkpoint(layer_module, **model_kwargs) if self.gradient_checkpoint else layer_module(**model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs
    
    def apply_final_layers(self, **model_kwargs):
        """根据剩余参数决定输出

        :param inputs: List[torch.Tensor], 默认顺序为[encoded_layers, conditional_emb]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        # 获取最后一层隐藏层的输出
        encoded_layers, conditional_emb = model_kwargs['encoded_layers'], model_kwargs.get('conditional_emb', None)
        sequence_output = encoded_layers[-1]
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加pool层
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        # 是否添加nsp
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        # 是否添加mlm
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm((mlm_hidden_state, conditional_emb))
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation('linear' if self.with_mlm is True else self.with_mlm)
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None
        
        outputs = [value for value in [encoded_layers, pooled_output, mlm_scores, nsp_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]

    def load_variable(self, state_dict, name, prefix='bert'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        variable = state_dict[name]
        if name in {
            f'{prefix}.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == f'{prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'cls.seq_relationship.weight':
            return variable.T
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        mapping = {
            'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias',
            'pooler.weight': f'{prefix}.pooler.dense.weight',
            'pooler.bias': f'{prefix}.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight',
            'mlmDecoder.bias': 'cls.predictions.decoder.bias'

        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.output.dense.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.output.dense.bias',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.output.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'output.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'output.LayerNorm.bias'
                            })

        if self.embedding_size != self.hidden_size:
            mapping.update({'embeddings.embedding_hidden_mapping_in.weight': f'{prefix}.encoder.embedding_hidden_mapping_in.weight',
                            'embeddings.embedding_hidden_mapping_in.bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias'})
        return mapping


class ALBERT(BERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])  # 取上述的第一行

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和BERT区别是始终使用self.encoderLayer[0]）；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """

        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i in range(self.num_hidden_layers):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            layer_module = self.encoderLayer[0]
            outputs = grad_checkpoint(layer_module, **model_kwargs) if self.gradient_checkpoint else layer_module(**model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] =  encoded_layers
        return model_kwargs


    def variable_mapping(self, prefix='albert'):
        mapping = {
            'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias',
            'embeddings.embedding_hidden_mapping_in.weight': f'{prefix}.encoder.embedding_hidden_mapping_in.weight',
            'embeddings.embedding_hidden_mapping_in.bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias',
            'pooler.weight': f'{prefix}.pooler.weight',
            'pooler.bias': f'{prefix}.pooler.bias',
            'nsp.weight': 'sop_classifier.classifier.weight',  # 用名字nsp来替换sop
            'nsp.bias': 'sop_classifier.classifier.bias',
            'mlmDense.weight': 'predictions.dense.weight',
            'mlmDense.bias': 'predictions.dense.bias',
            'mlmLayerNorm.weight': 'predictions.LayerNorm.weight',
            'mlmLayerNorm.bias': 'predictions.LayerNorm.bias',
            'mlmBias': 'predictions.bias',
            'mlmDecoder.weight': 'predictions.decoder.weight',
            'mlmDecoder.bias': 'predictions.decoder.bias'
        }
        i = 0
        prefix_i = f'{prefix}.encoder.albert_layer_groups.{i}.albert_layers.{i}.'
        mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.query.weight',
                        f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.query.bias',
                        f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.key.weight',
                        f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.key.bias',
                        f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.value.weight',
                        f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.value.bias',
                        f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight',
                        f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias',
                        f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.LayerNorm.weight',
                        f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.LayerNorm.bias',
                        f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ffn.weight',
                        f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ffn.bias',
                        f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ffn_output.weight',
                        f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ffn_output.bias',
                        f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'full_layer_layer_norm.weight',
                        f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'full_layer_layer_norm.bias'
                        })

        return mapping

    def load_variable(self, state_dict, name):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {
            'albert.embeddings.word_embeddings.weight',
            'predictions.bias',
            'predictions.decoder.weight',
            'predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == 'albert.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'sop_classifier.classifier.weight':
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)])

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和ALBERT区别是所有层权重独立）；这里就是调用BERT类的方法
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        return BERT.apply_main_layers(self, **model_kwargs)

    def variable_mapping(self, prefix='albert'):
        mapping = super().variable_mapping()
        prefix_0 = f'{prefix}.encoder.albert_layer_groups.0.albert_layers.0.'
        for i in range(1, self.num_hidden_layers):
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_0 + 'attention.query.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_0 + 'attention.query.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_0 + 'attention.key.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_0 + 'attention.key.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_0 + 'attention.value.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_0 + 'attention.value.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_0 + 'attention.dense.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_0 + 'attention.dense.bias',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_0 + 'attention.LayerNorm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_0 + 'attention.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_0 + 'ffn.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_0 + 'ffn.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_0 + 'ffn_output.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_0 + 'ffn_output.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_0 + 'full_layer_layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_0 + 'full_layer_layer_norm.bias'
                            })
        return mapping


class NEZHA(BERT):
    """华为推出的NAZHA模型；
    链接：https://arxiv.org/abs/1909.00204
    """
    def __init__(self, *args, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding, max_relative_position默认取64
        kwargs.update({'p_bias': 'typical_relative', 'max_relative_position': kwargs.get('max_relative_position', 64)})
        super(NEZHA, self).__init__(*args, **kwargs)


class RoFormer(BERT):
    """旋转式位置编码的BERT模型；
    链接：https://kexue.fm/archives/8265
    """
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary'})
        super(RoFormer, self).__init__(*args, **kwargs)
    
    def load_variable(self, state_dict, name, prefix='roformer'):
        return super().load_variable(state_dict, name, prefix)

    def variable_mapping(self, prefix='roformer'):
        mapping =  super().variable_mapping(prefix)
        del mapping['embeddings.position_embeddings.weight'] # 没有位置编码
        return mapping


class RoFormerV2(RoFormer):
    """RoFormerV2；
    改动：去掉bias，简化Norm，优化初始化等。目前初始化暂时还用的bert的初始化，finetune不受影响
    """
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm'})
        super(RoFormerV2, self).__init__(*args, **kwargs)
        if self.with_mlm:
            del self.mlmLayerNorm
            del self.mlmBias
            del self.mlmDense
            self.mlmDecoder.register_parameter('bias', None)

    def variable_mapping(self, prefix='roformer'):
        mapping = super().variable_mapping(prefix)
        mapping_new = {}
        for k, v in mapping.items():
            if (not re.search('bias|layernorm', k.lower())) and (not re.search('bias|layernorm', v.lower())):
                mapping_new[k] = v
        return mapping_new

    def apply_final_layers(self, **model_kwargs):
        """根据剩余参数决定输出
        """
        # 获取最后一层隐藏层的输出
        encoded_layers = model_kwargs['encoded_layers']
        sequence_output = encoded_layers[-1]
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加mlm
        if self.with_mlm:
            mlm_scores = self.mlmDecoder(sequence_output)
        else:
            mlm_scores = None
        
        outputs = [value for value in [encoded_layers, mlm_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm', 'normalization': 'softmax_plus'})
        super().__init__(*args, **kwargs)

        layer = self.GAU_Layer(**kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
    
    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        return self.load_embeddings(variable) if name in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'} else variable

    def variable_mapping(self, prefix=''):
        '''在convert脚本里已经把key转成bert4torch可用的
        '''
        return {k: k for k, _ in self.named_parameters()}

    class GAU_Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gau = GatedAttentionUnit(**kwargs)
            self.dropout1 = nn.Dropout(kwargs.get('dropout_rate'))
            self.layerNorm1 = LayerNorm(**kwargs)
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, **model_kwargs):
            gau_hidden_states = self.gau(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(gau_hidden_states)
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs

    
class ELECTRA(BERT):
    """Google推出的ELECTRA模型；
    链接：https://arxiv.org/abs/2003.10555
    """
    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = get_activation('sigmoid') if self.with_discriminator is True else get_activation(self.with_discriminator)

    def apply_final_layers(self, **model_kwargs):
        hidden_states = super().apply_final_layers(**model_kwargs)  # 仅有hidden_state一项输出
        if self.with_discriminator:
            logit = self.dense_act(self.dense(hidden_states))
            return [hidden_states, self.dense_prediction_act(self.dense_prediction(logit))]
        else:
            return hidden_states

    def load_variable(self, state_dict, name):
        # 加载单个变量的函数
        return super().load_variable(state_dict, name, prefix='electra')

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix='electra')
        mapping.update({'dense.weight': 'discriminator_predictions.dense.weight', 
                        'dense.bias': 'discriminator_predictions.dense.bias',
                        'dense_prediction.weight': 'discriminator_predictions.dense_prediction.weight',
                        'dense_prediction.bias': 'discriminator_predictions.dense_prediction.bias'}
                        )
        for del_key in ['pooler.weight', 'pooler.bias', 'nsp.weight', 'nsp.bias', 'mlmDense.weight', 'mlmDense.bias', 
                        'mlmLayerNorm.weight', 'mlmLayerNorm.bias', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias']:
            del mapping[del_key]

        return mapping


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""
    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)

    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if ('LayerNorm.weight' in v) or ('LayerNorm.bias' in v):
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='ernie'):
        return super().load_variable(state_dict, name, prefix=prefix)


class DebertaV2(BERT):
    '''DeBERTaV2: https://arxiv.org/abs/2006.03654, https://github.com/microsoft/DeBERTa；
       这里使用的是IEDEA的中文模型：https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese；
       和transformers包中的区别：
       1）原始使用的StableDropout替换成了nn.Dropout；
       2）计算attention_score时候用的XSoftmax替换成了F.softmax；
       3）未实现（认为对结果无影响）：Embedding阶段用attention_mask对embedding的padding部分置0；
       4）未实现（认为对结果无影响）：计算attention_score前的attention_mask从[btz, 1, 1, k_len]转为了[btz, 1, q_len, k_len]。
    '''
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'deberta_v2'})  # 控制在Embedding阶段不生成position_embedding
        super(DebertaV2, self).__init__(*args, **kwargs)
        
        # Encoder中transformer_block前的其他网络结构
        self.relative_attention = kwargs.get("relative_attention", True)
        self.conv = ConvLayer(**kwargs) if kwargs.get("conv_kernel_size", 0) > 0 else None
        
        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.weight = self.encoderLayer[0].multiHeadAttention.layernorm.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.bias = self.encoderLayer[0].multiHeadAttention.layernorm.bias

    def apply_main_layers(self, **model_kwargs):
        """DebertaV2：主要区别是第0层后，会通过卷积层
        """

        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.encoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = grad_checkpoint(layer_module, **model_kwargs) if self.gradient_checkpoint else layer_module(**model_kwargs)
            model_kwargs.update(outputs)
            # 第0层要经过卷积
            if l_i == 0 and self.conv is not None:
                model_kwargs['hidden_states'] = self.conv(encoded_layers[0], model_kwargs['hidden_states'], model_kwargs['attention_mask'].squeeze(1).squeeze(1))
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                encoded_layers.append(model_kwargs['hidden_states'])
        if not self.output_all_encoded_layers:
            encoded_layers.append(model_kwargs['hidden_states'])
        model_kwargs['encoded_layers'] =  encoded_layers
        return model_kwargs

    def variable_mapping(self):
        mapping = super(DebertaV2, self).variable_mapping(prefix='deberta')
        mapping.update({'mlmDecoder.weight': 'deberta.embeddings.word_embeddings.weight',
                        'mlmDecoder.bias': 'cls.predictions.bias',
                        'encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'deberta.encoder.rel_embeddings.weight',
                        'encoderLayer.0.multiHeadAttention.layernorm.weight': 'deberta.encoder.LayerNorm.weight',
                        'encoderLayer.0.multiHeadAttention.layernorm.bias': 'deberta.encoder.LayerNorm.bias',
                        'conv.conv.weight': 'deberta.encoder.conv.conv.weight',
                        'conv.conv.bias': 'deberta.encoder.conv.conv.bias',
                        'conv.LayerNorm.weight': 'deberta.encoder.conv.LayerNorm.weight',
                        'conv.LayerNorm.bias': 'deberta.encoder.conv.LayerNorm.bias'})
        for del_key in ['nsp.weight', 'nsp.bias', 'embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='deberta'):
        return super().load_variable(state_dict, name, prefix=prefix)


class UIE(BERT):
    '''官方项目：https://github.com/universal-ie/UIE；
       参考项目：https://github.com/heiheiyoyo/uie_pytorch
    '''
    @delete_arguments('with_nsp', 'with_mlm')
    def __init__(self, *args, **kwargs):
        super(UIE, self).__init__(*args, **kwargs)
        hidden_size = self.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        if kwargs.get('sigmoid', True):
            self.sigmoid = nn.Sigmoid()

        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output+task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def apply_final_layers(self, **model_kwargs):
        hidden_states = super().apply_final_layers(**model_kwargs)  # 仅有hidden_state一项输出
        sequence_output = hidden_states[0] if isinstance(hidden_states, (tuple, list)) else hidden_states

        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits) if hasattr(self, 'sigmoid') else start_logits
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits) if hasattr(self, 'sigmoid') else end_logits

        if isinstance(hidden_states, (tuple, list)):
            return hidden_states + [start_prob, end_prob]
        else:
            return hidden_states, start_prob, end_prob

    def variable_mapping(self):
        mapping = super(UIE, self).variable_mapping()
        mapping.update({'linear_start.weight': 'linear_start.weight',
                        'linear_start.bias': 'linear_start.bias',
                        'linear_end.weight': 'linear_end.weight',
                        'linear_end.bias': 'linear_end.bias'})
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping


class Encoder(BERT):
    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = kwargs.get('src_vocab_size', kwargs['vocab_size'])
        super().__init__(*args, **kwargs)
        # encoder需要返回encoder_attention_mask
        self.encoder_attention_mask = None
    
    def forward(self, *inputs, **model_kwargs):
        """因为encoder需要返回encoder_attention_mask，因此这里从新定义一下，多返回一个参数
        """
        # 返回model_kwargs方便解析attention_mask
        outputs, model_kwargs = super().forward(*inputs, use_states=True, **model_kwargs)
        # return: [encoder_hidden_states, encoder_attention_mask]
        return ([outputs] if isinstance(outputs, torch.Tensor) else outputs) + [model_kwargs['attention_mask']]


class Decoder(LM_Mask, BERT):
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, with_lm=True, tie_emb_prj_weight=False, logit_scale=True, **kwargs):
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True  # 标记是decoder
        super().__init__(*args, **kwargs)
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.with_lm = with_lm

        # 从hidden_states映射到logit
        if self.with_lm:
            self.final_dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            # decoder底层的embedding和顶层的全连接共享
            # [True]: fudan_bart和uer_t5的t5, [False]: mt5和t5_pegasus
            if tie_emb_prj_weight:
                self.final_dense.weight = self.embeddings.word_embeddings.weight
            if logit_scale:  # T5默认会有logit_scale, bart默认没有，所以bart要传入false
                self.x_logit_scale = (self.hidden_size ** -0.5)
            else:
                self.x_logit_scale = 1.

    def apply_main_layers(self, **model_kwargs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块；
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        decoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.decoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = grad_checkpoint(layer_module, **model_kwargs) if self.gradient_checkpoint else layer_module(**model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)

            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        model_kwargs['decoded_layers'] = decoded_layers
        return model_kwargs
    
    def apply_final_layers(self, **model_kwargs):
        outputs = []
        hidden_states = model_kwargs['decoded_layers'][-1]  # outputs为decoder顶层的hidden_states [btz, seq_len, hdsz]
        outputs.append(hidden_states)
        if self.with_lm:
            logits = self.final_dense(hidden_states) * self.x_logit_scale  # outputs为[btz, seq_len, vocab_size]的logits
            activation = get_activation('linear' if self.with_lm is True else self.with_lm)  # 添加激活，一般是线性激活或softmax
            logits = activation(logits)
            outputs.append(logits)
        return outputs

    def variable_mapping(self, prefix='bert'):
        raw_mapping = super().variable_mapping(prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        return mapping


class Transformer(BERT_BASE):
    '''encoder-decoder结构'''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_emb_src_tgt_weight=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)

        # encoder
        self.encoder = Encoder(*args, **kwargs)

        # decoder
        self.decoder = Decoder(*args, add_cross_attention=True, **kwargs)

        if tie_emb_src_tgt_weight:
            # encoder和decoder的embedding权重共享
            assert self.encoder.vocab_size == self.decoder.vocab_size, "To share word embedding, the vocab size of src/tgt shall be the same."
            self.encoder.embeddings.word_embeddings.weight = self.decoder.embeddings.word_embeddings.weight

    def forward(self, *inputs):
        inputs = self.args_segmentate(inputs)
        encoder_input, decoder_input = inputs[:2]

        # encoder
        encoder_hidden_states, encoder_attention_mask = self.encoder(encoder_input)

        # decoder
        decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])
        return [encoder_hidden_states] + decoder_outputs  # 输出encoder_hidden_state和decoder_hidden_state，以应对一些多任务情况


class BART(Transformer):
    '''encoder-decoder结构'''
    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        kwargs['logit_scale'] = kwargs.get('logit_scale', False)
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(BART, self).__init__(*args, tie_emb_src_tgt_weight=tie_emb_src_tgt_weight, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

    def load_variable(self, state_dict, name, prefix=''):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {
            'shared.weight',
            'encoder.embed_tokens.weight',
            'decoder.embed_tokens.weight',
        }:
            return self.load_embeddings(variable)
        elif name in {'encoder.embed_positions.weight', 'decoder.embed_positions.weight'}:
            return self.load_pos_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {
            'encoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'encoder.embed_tokens.weight',
            'encoder.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
            'encoder.embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight',
            'encoder.embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias',
            'decoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'decoder.embed_tokens.weight',
            'decoder.embeddings.position_embeddings.weight': 'decoder.embed_positions.weight',
            'decoder.embeddings.layerNorm.weight': 'decoder.layernorm_embedding.weight',
            'decoder.embeddings.layerNorm.bias': 'decoder.layernorm_embedding.bias',
        }
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight',
                f'encoder.encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias',
                f'encoder.encoderLayer.{i}.layerNorm1.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight',
                f'encoder.encoderLayer.{i}.layerNorm1.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight',
                f'encoder.encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight',
                f'encoder.encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias',
                f'encoder.encoderLayer.{i}.layerNorm2.weight': f'encoder.layers.{i}.final_layer_norm.weight',
                f'encoder.encoderLayer.{i}.layerNorm2.bias': f'encoder.layers.{i}.final_layer_norm.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.layerNorm1.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm1.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias',
                f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight',
                f'decoder.decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias',
                f'decoder.decoderLayer.{i}.layerNorm3.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm3.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight',
                f'decoder.decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight',
                f'decoder.decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias',
                f'decoder.decoderLayer.{i}.layerNorm2.weight': f'decoder.layers.{i}.final_layer_norm.weight',
                f'decoder.decoderLayer.{i}.layerNorm2.bias': f'decoder.layers.{i}.final_layer_norm.bias'
                })

        return mapping


class T5_Encoder(Encoder):
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version, 
                       'bias': False, 'norm_mode': 'rmsnorm'})  # p_bias来控制embedding阶段无pos_embedding，t5不使用bias，并且使用rmsnorm
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # t5的layernorm都在前面，因此重新定义了下
        layer = T5Layer(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                      'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, **model_kwargs):
        hidden_states = super().apply_final_layers(**model_kwargs)
        return self.dropout(self.final_layer_norm([hidden_states]))

    def load_variable(self, state_dict, name, prefix=''):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {'encoder.embed_tokens.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight',
                   f'{prefix}encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'{prefix}final_layer_norm.weight': 'encoder.final_layer_norm.weight'}
        for i in range(self.num_hidden_layers):
            mapping.update({
                f'{prefix}encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'{prefix}encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'{prefix}encoderLayer.{i}.layerNorm1.weight': f'encoder.block.{i}.layer.0.layer_norm.weight',
                f'{prefix}encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight',
                f'{prefix}encoderLayer.{i}.layerNorm2.weight': f'encoder.block.{i}.layer.1.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight',
                                f'{prefix}encoderLayer.{i}.feedForward.intermediateDense1.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'})
        return mapping
    

class T5_Decoder(Decoder):
    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version,
                       'bias': False, 'norm_mode': 'rmsnorm'})  # p_bias来控制embedding阶段无pos_embedding，t5不使用bias，并且使用rmsnorm
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        # t5的layernorm都在前面，因此重新定义了下
        layer = T5Layer(is_decoder=True, **self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                                       'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        
        # 把第二层后的相对位置编码的权重绑定到第一层上，变相实现仅由第一层计算
        for i in range(1, self.num_hidden_layers):
            self.decoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.decoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, **model_kwargs):
        # 这里的encoded_layers没有改成decoded_layers是想使用super()
        last_hidden_states = model_kwargs['decoded_layers'][-1]
        model_kwargs['decoded_layers'][-1] = self.dropout(self.final_layer_norm([last_hidden_states]))  # 在转logit前把最后一层的hidden_states加layernorm
        return super().apply_final_layers(**model_kwargs)

    def load_variable(self, state_dict, name, prefix=''):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {f'decoder.embed_tokens.weight', 'lm_head.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        # 查看check_point发现'shared.weight'
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'decoder.embed_tokens.weight',
                   f'{prefix}decoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                   f'{prefix}final_layer_norm.weight': 'decoder.final_layer_norm.weight',
                   f'{prefix}final_dense.weight': 'lm_head.weight'}

        for i in range(self.num_hidden_layers):
            mapping.update({
                f'{prefix}decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.block.{i}.layer.0.SelfAttention.q.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.block.{i}.layer.0.SelfAttention.k.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.block.{i}.layer.0.SelfAttention.v.weight',
                f'{prefix}decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.block.{i}.layer.0.SelfAttention.o.weight',
                f'{prefix}decoderLayer.{i}.layerNorm1.weight': f'decoder.block.{i}.layer.0.layer_norm.weight',

                f'{prefix}decoderLayer.{i}.crossAttention.q.weight': f'decoder.block.{i}.layer.1.EncDecAttention.q.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.k.weight': f'decoder.block.{i}.layer.1.EncDecAttention.k.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.v.weight': f'decoder.block.{i}.layer.1.EncDecAttention.v.weight',
                f'{prefix}decoderLayer.{i}.crossAttention.o.weight': f'decoder.block.{i}.layer.1.EncDecAttention.o.weight',
                f'{prefix}decoderLayer.{i}.layerNorm3.weight': f'decoder.block.{i}.layer.1.layer_norm.weight',

                f'{prefix}decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight',
                f'{prefix}decoderLayer.{i}.layerNorm2.weight': f'decoder.block.{i}.layer.2.layer_norm.weight',
                })

            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight',
                                f'{prefix}decoderLayer.{i}.feedForward.intermediateDense1.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'})
        return mapping


class T5(Transformer):
    """Google的T5模型（Encoder-Decoder）"""
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args,  tie_emb_src_tgt_weight=True, **kwargs):
        super(T5, self).__init__(*args, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

        # encoder
        self.encoder = T5_Encoder(*args, **kwargs)

        # decoder
        kwargs['add_cross_attention'] = True
        self.decoder = T5_Decoder(*args, **kwargs)

    def load_variable(self, state_dict, name, prefix=''):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = self.encoder.variable_mapping(prefix='encoder.')
        mapping.update(self.decoder.variable_mapping(prefix='decoder.'))
        if self.tie_emb_src_tgt_weight:
            mapping.update({'encoder.embeddings.word_embeddings.weight': 'shared.weight',
                            'decoder.embeddings.word_embeddings.weight': 'shared.weight'})
        return mapping


class GPT(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    Todo: 理论上gpt系列应该从Decoder继承，自动实现is_decoder=True，且使用decoderLayer
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        """GPT的embedding是token、position、segment三者embedding之和，跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT, self).__init__(*args, is_decoder=True, **kwargs)
        del self.embeddings.layerNorm
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT, self).load_variable(state_dict, name, prefix='gpt')

    def variable_mapping(self):
        # 映射到GPT权重格式
        mapping =  super(GPT, self).variable_mapping(prefix='gpt')
        return mapping


class GPT2(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        """GPT2的embedding是token、position两者embedding之和。
           1、跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           2、bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT2, self).__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = self.Gpt2Layer(is_decoder=True, **self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                                              'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=kwargs.get('bias', True))
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False) 
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(self.LayerNormFinal([hidden_state]))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2, self).load_variable(state_dict, name, prefix='gpt2')

    def variable_mapping(self):
        # 映射到GPT权重格式
        mapping = super(GPT2, self).variable_mapping(prefix='gpt2')
        mapping.update({'LayerNormFinal.weight': 'gpt2.LayerNormFinal.weight',
                        'LayerNormFinal.bias': 'gpt2.LayerNormFinal.bias'})
        return mapping
    
    class Gpt2Layer(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
            x = self.layerNorm1((hidden_states, conditional_emb))
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value)
            hidden_states = hidden_states + self.dropout1(self_attn_output[0])

            x = self.layerNorm2((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)

            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs


class GPT2_ML(LM_Mask, BERT):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    看完ckpt中的key，和GPT的区别是embedding后也有layernorm，和bert的区别是第二个跳跃链接的输入是在layernorm前，bert是在之后
    """
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layer = self.Gpt2MlLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, 
                                 is_dropout=self.is_dropout, conditional_size=self.conditional_size, is_decoder=True)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix='gpt2_ml')

    def variable_mapping(self):
        # 映射到GPT2权重格式
        mapping =  super(GPT2_ML, self).variable_mapping(prefix='gpt2_ml')
        return mapping

    class Gpt2MlLayer(BertLayer):
        '''未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        '''
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
            hidden_states = hidden_states + self.dropout1(self_attn_output[0])
            x = self.layerNorm1((hidden_states, conditional_emb))

            ffn_output = self.feedForward(x)
            # bert的第二个跳跃连接的输入1是经过了multiHeadAttention+layerNorm1的hidden_states, 即这里的x
            # gpt2_ml的第二个跳跃连接的输入1是经过了multiHeadAttention的hidden_states, 不加layerNorm1
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs


class LLaMA(LM_Mask, BERT):
    '''LLaMA
    链接: https://github.com/facebookresearch/llama
    改动：模型结构和gpt2类似，去掉bias，简化Norm, feedForward不同
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': True, 'bias': False, 'norm_mode': 'rmsnorm', 'is_decoder': True})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = self.TransformerBlock(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                                    'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, norm_mode=kwargs['norm_mode'], bias=kwargs['bias'])
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False, device=kwargs['skip_init']) 
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))
        # 修改feedword
        for layer in self.encoderLayer:
            layer.feedForward = self.FeedForward(self.hidden_size, self.hidden_size*4, **kwargs)

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(self.LayerNormFinal([hidden_state]))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(LLaMA, self).load_variable(state_dict, name, prefix='llama')

    def variable_mapping(self, prefix='llama'):
        # 映射到权重格式
        mapping = super(LLaMA, self).variable_mapping(prefix=prefix)
        mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight',
                        'dense.weight': f'{prefix}.dense.weight'})
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.feedForward.intermediateDense2.weight': prefix_i + 'intermediate2.dense.weight'})
        return mapping
    
    class TransformerBlock(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            del self.dropout1
            del self.dropout2

        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
            # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
            x = self.layerNorm1((hidden_states, conditional_emb))
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
            hidden_states = hidden_states + self_attn_output[0]

            x = self.layerNorm2((hidden_states, conditional_emb))
            hidden_states = hidden_states +  self.feedForward(x)
            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs
        
    class FeedForward(nn.Module):
        '''FeedForward和Bert的不一致，Bert只有两个全连接'''
        def __init__(self, dim: int, hidden_dim: int, multiple_of: int, **kwargs):
            super().__init__()
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            self.intermediateDense = nn.Linear(dim, hidden_dim, bias=False, device=kwargs['skip_init'])
            self.outputDense = nn.Linear(hidden_dim, dim, bias=False, device=kwargs['skip_init'])
            self.intermediateDense2 = nn.Linear(dim, hidden_dim, bias=False, device=kwargs['skip_init'])

        def forward(self, x):
            return self.outputDense(F.silu(self.intermediateDense(x)) * self.intermediateDense2(x))


class GLM(LM_Mask, BERT):
    '''GLM: https://github.com/THUDM/GLM
    Unilm设计，可定义为GLM(UniLM_MASK, BERT)但是要求传入segement_ids比较麻烦，这里继承LM_MASK并使用get_masks()重新构造attention_mask
    模型结构特点：
    1）rotary使用的updown+position_encoding_2d
    2）qkv合并成一个权重convert时不是concat在一起的
    3）attention_mask类似于Unilm，最后一个token仅能访问之前的，之前的tokens可以互相访问
    4）跳跃连接有权重设计
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': True, 'is_decoder': True})
        super().__init__(*args, **kwargs)
        self.bos_token_id, self.mask_token_id, self.gmask_token_id = kwargs['bos_token_id'], kwargs['mask_token_id'], kwargs['gmask_token_id']
        self.position_encoding_2d = kwargs.get('position_encoding_2d', True)
        del self.embeddings.layerNorm
        layer = self.GLMBlock(**self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                                            'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'num_hidden_layers', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = torch.nn.LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12))
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False, device=kwargs['skip_init'])
        self.final_activation = get_activation(kwargs.get('final_activation', 'linear'))

    def load_variable(self, state_dict, name, prefix='transformer'):
        """加载单个变量的函数, 这里的名称均为映射前的
        """
        variable = state_dict[name]
        if name in {f'{prefix}.embeddings.word_embeddings.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable
        
    def variable_mapping(self, prefix='transformer'):
        # 映射到权重格式
        mapping = {
            'LayerNormFinal.weight': "transformer.final_layernorm.weight",
            'LayerNormFinal.bias': "transformer.final_layernorm.bias",
            'dense.weight': "lm_head.weight",
            'embeddings.word_embeddings.weight': 'transformer.word_embeddings.weight'}

        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layers.%d.' % i
            mapping.update({
                f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'input_layernorm.weight',
                f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'input_layernorm.bias',
                f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'post_attention_layernorm.weight',
                f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'post_attention_layernorm.bias',
                f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight',
                f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias',
                f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight',
                f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias',
                f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight',
                f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias',
                f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight',
                f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias',
                f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'mlp.dense_h_to_4h.weight',
                f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'mlp.dense_h_to_4h.bias',
                f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'mlp.dense_4h_to_h.weight',
                f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'mlp.dense_4h_to_h.bias',
                })
        return mapping

    def get_masks(self, attention_mask, context_lens, prepad_lens):
        '''调整mask使得在content_lens前是bi_attention'''
        for i, (prepad_len, context_len) in enumerate(zip(prepad_lens, context_lens)):
            attention_mask[i, :, :, prepad_len:context_len] = 1
        return attention_mask
        
    def get_position_ids(self, position_ids, seq_len, context_lens, mask_positions, prepad_lens, gmask=False):
        '''不使用cache时候的postion_ids'''
        if position_ids.shape[0] == 1:
            position_ids = position_ids.repeat(len(context_lens), 1)
        if self.position_encoding_2d:
            # 初始版本中这里也有not gmask
            for i, context_length in enumerate(context_lens):
                position_ids[i, context_length:] = mask_positions[i] - prepad_lens[i]
            block_position_ids = [torch.cat((torch.zeros(context_len, dtype=torch.long).to(position_ids),
                                            torch.arange(seq_len-context_len, dtype=torch.long).to(position_ids) + 1)) for context_len in context_lens]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            if not gmask:
                for i, context_length in enumerate(context_lens):
                    position_ids[context_length:] = mask_positions[i] - prepad_lens[i]
        return position_ids

    def prepare_inputs(self, *inputs, **model_kwargs):
        '''对attention_mask(参考unilm方式)和position_ids做处理'''
        token_ids = model_kwargs['past_token_ids'] if model_kwargs.get('past_token_ids') is not None else inputs[0]
        mask_token = self.mask_token_id if self.mask_token_id in token_ids else self.gmask_token_id  # 倒数第2位
        use_gmask = False if self.mask_token_id in token_ids else True
        position_ids = model_kwargs['position_ids']
        device = position_ids.device
        seqs = token_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]
        context_lens = [seq.index(self.bos_token_id) for seq in seqs]  # bos_token_id是倒数第一位
        seq_len = token_ids.shape[1]

        # 1）generation阶段use_states=True且step>0的时候(用cache)
        if model_kwargs.get('use_states', False) and (model_kwargs.get('step') > 0):
            if self.position_encoding_2d:  # [btz, 2, 1]
                position_ids = torch.tensor([[mask_position, seq_len - context_len] for mask_position, context_len in
                                            zip(mask_positions, context_lens)], dtype=torch.long, device=device).unsqueeze(-1)
            else:  # [btz, 1]
                position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long, device=device).unsqueeze(-1)
            model_kwargs['position_ids'] = position_ids
        # 1）train阶段；2）generation阶段use_states=False；3）use_states=True且step=0的时候
        else:
            prepad_lens = [(ts[:l]==3).sum().item() for l, ts in zip(context_lens, token_ids)]
            model_kwargs['attention_mask'] = self.get_masks(model_kwargs['attention_mask'], context_lens, prepad_lens)
            model_kwargs['position_ids'] = self.get_position_ids(position_ids, seq_len, context_lens, mask_positions, prepad_lens, gmask=use_gmask)
        return model_kwargs

    def apply_embeddings(self, *inputs, **model_kwargs):
        model_kwargs = super().apply_embeddings(*inputs, **model_kwargs)
        model_kwargs = self.prepare_inputs(*inputs, **model_kwargs)
        return model_kwargs
    
    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        logit = self.dense(self.LayerNormFinal(hidden_state))
        return self.final_activation(logit)
    
    class GLMBlock(BertLayer):
        '''顺序：LN --> Att --> Add --> LN --> FFN --> Add'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_hidden_layers = kwargs['num_hidden_layers']
            hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-5)
            self.layerNorm1 = torch.nn.LayerNorm(hidden_size, eps=eps)
            self.layerNorm2 = torch.nn.LayerNorm(hidden_size, eps=eps)
            del self.dropout1
            del self.dropout2

        def forward(self, hidden_states=None, attention_mask=None, past_key_value=None, **model_kwargs):
            # 和bert区别有两点，一个是有alpha, 还有一个是跳跃链接用的是经过了layernorm后的
            x = self.layerNorm1(hidden_states)
            alpha = (2 * self.num_hidden_layers) ** 0.5
            self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
            hidden_states = x * alpha + self_attn_output[0]

            x = self.layerNorm2(hidden_states)
            hidden_states = x *alpha +  self.feedForward(x)

            if self.is_decoder:
                model_kwargs['past_key_value'] = self_attn_output[-1]
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs


class Transformer_XL(BERT):
    '''构建transformer-xl模型, 已加载；
    项目: https://github.com/kimiyoung/transformer-xl；
    不同点:  
    1) 简化了原有的AdaptiveEmbedding(可选)和未使用ProjectedAdaptiveLogSoftmax, 直接输出last_hidden_state；
    2) mems修改了transformer中初始化为zero_tensor, 改为包含最后一层, 原项目初始化为empty_tensor；
    3) SinusoidalPositionEncoding一般是sincos间隔排列, 这里是先sin后cos；
    4) attention_mask在multi_attn中使用中使用1e30来替代原来的1000。
    '''
    @delete_arguments('with_pool', 'with_nsp', 'with_mlm')
    @insert_arguments(with_lm=False)
    def __init__(self, *args, mem_len=0, same_length=False, clamp_len=-1, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding
        kwargs.update({'p_bias': 'other_relative'})
        super().__init__(*args, **kwargs)
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        self.attn_type = kwargs.get('attn_type', 0)

        # embedding
        if kwargs.get('adaptive_embedding'):
            cutoffs, div_val, sample_softmax = kwargs.get('cutoffs', []), kwargs.get('div_val', 1), kwargs.get('sample_softmax', False)
            self.embeddings = AdaptiveEmbedding(self.vocab_size, self.embedding_size, self.hidden_size, cutoffs, div_val, sample_softmax)
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_embeddings = XlnetPositionsEncoding(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # 每层自己的r_w_bias和r_r_bias，还是公用
        if not kwargs.get('untie_r'):
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局内容偏置
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局位置偏置
            if self.segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))  # 全局segment偏置
        else:
            self.r_w_bias, self.r_r_bias = None, None
            self.r_s_bias = None

        # transformer block
        layer = XlnetLayer(r_s_bias=None, **self.get_kw('hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 'intermediate_size', 
                                                        'hidden_act', 'is_dropout', 'conditional_size', 'r_w_bias', 'r_r_bias', **kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])

        # 映射
        if self.with_lm:
            self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

    def init_mems(self, bsz):
        '''初始化mems, 用于记忆mlen的各层隐含层状态'''
        if isinstance(self.mem_len, (int, float)) and (self.mem_len > 0):
            mems = []
            param = next(self.parameters())
            for _ in range(self.num_hidden_layers+1):
                empty = torch.zeros(bsz, self.mem_len, self.hidden_size, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mlen, qlen):
        '''更新mems'''
        # does not deal with None
        if self.mems is None:
            return None
        # mems is not None
        assert len(hids) == len(self.mems), "len(hids) != len(mems)"
        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        self.mems = new_mems

    def relative_positional_encoding(self, qlen, klen, device):
        # 生成pos_emb, 这里使用sincos的位置编码，为了和xlnet入参一致
        pos_seq = torch.arange(klen-1, -1, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.dropout(self.pos_embeddings(pos_seq))  # 用word_emb的dropout
        return pos_emb

    def create_mask(self, word_emb, qlen, klen, mlen):
        # 修改attention_mask, mlen可以全部访问，q_len只能访问<=t时刻的, mask和Unilm类似，但是Unilm是靠segement_ids来控制
        if self.same_length:  # 只能访问前面固定长度
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = 1-(torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len)).byte() # -1
        else:
            attention_mask = torch.tril(word_emb.new_ones(qlen, klen), diagonal=mlen).byte()  # [q_len, k_len], 下三角为1矩阵
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def apply_embeddings(self, *inputs, **model_kwargs):
        '''接受的inputs输入: [token_ids, segment_ids], 暂不支持条件LayerNorm输入'''
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'

        self.mems = self.init_mems(inputs[0].size(0))  # 生成mems
        # 精简后embeddings中只计算word_emdedding
        word_emb = self.dropout(self.embeddings(inputs[0]))
        index_ = 1
        btz, qlen = inputs[0].shape[:2]  # query长度
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen
        # 相对位置编码
        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)
        # segment embedding
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            if mlen > 0:
                mem_pad = torch.zeros([btz, mlen], dtype=torch.long, device=word_emb.device)
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            # `1` indicates not in the same segment [qlen x klen x bsz]
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()
            index_ += 1
        else:
            segment_ids = None

        if self.attn_type in {'uni', 0}:  # 兼容transformer_xl的设置: 0
            non_tgt_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == 'bi':
            attention_mask = (inputs[0] != self.pad_token_id).long().unsqueeze(1).unsqueeze(2)
            non_tgt_mask = torch.eye(qlen).to(attention_mask)[None, None, :, :]
            non_tgt_mask = ((1 - attention_mask - non_tgt_mask) <= 0).long()
        model_kwargs.update({'hidden_states': word_emb, 'segment_ids': segment_ids, 'pos_emb': pos_emb, 
                             'attention_mask': non_tgt_mask})
        return model_kwargs

    def apply_main_layers(self, **model_kwargs):
        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.encoderLayer):
            mems_i = None if self.mems is None else self.mems[l_i]
            model_kwargs['mems_i'] = mems_i
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = grad_checkpoint(layer_module, **model_kwargs) if self.gradient_checkpoint else layer_module(**model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)
            encoded_layers.append(hidden_states)
        
        # 原实现中word_emb, pos_emb和core_out(hidden_states)使用同一个dropout
        hidden_states = self.dropout(hidden_states)
        qlen = hidden_states.size(1)  # query长度
        mlen = self.mems[0].size(0) if self.mems is not None else 0
        self._update_mems(encoded_layers, mlen, qlen)
        
        if not self.output_all_encoded_layers:
            # 不返回所有层，即返回顶层
            encoded_layers = encoded_layers[:1] + [hidden_states]
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs
    
    def load_variable(self, state_dict, name, prefix=''):
        # 这里由于预训练模型使用了AdapterEmbedding，因此暂不支持
        if (self.keep_tokens is not None) or (self.compound_tokens is not None):
            raise ValueError('Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL')
        return state_dict[name]

    def variable_mapping(self, prefix=''):
        return {k:k for k, v in self.named_parameters()}


class XLNET(Transformer_XL):
    '''构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入；
       接受的inputs输入: [token_ids, segment_ids]
    '''
    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)
    
    def relative_positional_encoding(self, qlen, klen, device):
        # 生成pos_emb, 这里使用sincos的位置编码, transformer_xl里面有-1
        if self.attn_type == 'bi':
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.") 

        # 前向的emb
        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)

        # 双向数据
        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb

        pos_emb = self.dropout(pos_emb)  # 用word_emb的dropout
        return pos_emb

    def apply_final_layers(self, **model_kwargs):
        hidden_state = super().apply_final_layers(**model_kwargs)
        if self.with_lm:
            return [hidden_state, self.dense(hidden_state)]
        else:
            return hidden_state

    def load_variable(self, state_dict, name, prefix='transformer'):
        # 加载单个变量的函数
        variable = state_dict[name]
        if name in {f'{prefix}.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        elif re.search('rel_attn\.(q|k|v|r)$', name):
            return variable.reshape(variable.shape[0], -1).T
        # elif re.search('rel_attn\.(o|seg_embed)$', name):
        elif re.search('rel_attn\.(o)$', name):
            return variable.reshape(variable.shape[0], -1)
        else:
            return variable

    def variable_mapping(self, prefix='transformer'):
        mapping = {
            'embeddings.weight': f'{prefix}.word_embedding.weight',
            'dense.weight': 'lm_loss.weight',
            'dense.bias': 'lm_loss.bias',
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'rel_attn.o',
                            f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r',
                            f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias',
                            f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias',
                            # f'encoderLayer.{i}.multiHeadAttention.seg_embed.weight': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed',
                            f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'rel_attn.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'rel_attn.layer_norm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'ff.layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'ff.layer_norm.bias'
                            })
        return mapping


def build_transformer_model(config_path=None, checkpoint_path=None, model='bert', application='encoder', **kwargs):
    """根据配置文件构建模型，可选加载checkpoint权重

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
    :param add_trainer: bool, 指定从BaseModel继承, 若build_transformer_model后需直接compile()、fit()需设置为True, 默认为None
    :param compound_tokens: 扩展Embedding, 默认为None
    :param residual_attention_scores: bool, Attention矩阵加残差, 默认为False
    :param ignore_invalid_weights: bool, 允许跳过不存在的权重, 默认为False
    :param keep_hidden_layers: 保留的hidden_layer层的id, 默认为None表示全部使用
    :param hierarchical_position: 是否层次分解位置编码, 默认为None表示不使用
    :param gradient_checkpoint: bool, 是否使用gradient_checkpoint, 默认为False
    :param skip_init: bool, 是否初始化
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
    configs['skip_init'] = 'meta' if configs.get('skip_init') is True else 'cpu'
    configs['add_trainer'] = configs.get('add_trainer', False)

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
    }

    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            configs['version'] = model
    elif isinstance(model, type) and issubclass(model, BERT_BASE): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('"model" args type should be string or nn.Module')

    # 使用 lm/unilm
    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5', ]:
        raise ValueError(f'"{model}" model can not be used as "{application}" application.\n')
    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    # 动态继承BaseModel
    if configs['add_trainer']:
        class MyModel(MODEL, BaseModel): 
            pass
        MODEL = MyModel

    # 生成网络结构
    transformer = MODEL(**configs)
    transformer.apply(transformer.init_model_weights)  # 初始化权重

    # 预训练模型是否已量化, 加载量化后的权重使用，如果是加载原权重再自行量化这里不需要甚至恶
    if configs.get('quantization_bit') is not None:
        bits = configs.pop('quantization_bit')
        transformer = transformer.half().quantize(bits=bits, **configs)

    # 权重加载
    if checkpoint_path is not None:
        verbose = not configs.get('ignore_invalid_weights', False)
        transformer.load_weights_from_pytorch_checkpoints(checkpoint_path, verbose=verbose)
    transformer.configs = transformer.config = configs
    return transformer