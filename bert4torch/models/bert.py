import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from bert4torch.models.base import PreTrainedModel
from bert4torch.models.modeling_utils import old_checkpoint
from bert4torch.layers import LayerNorm, BertEmbeddings, TRANSFORMER_BLOCKS
from bert4torch.layers import LayerNorm, BertEmbeddings, BertLayer, BlockIdentity
from bert4torch.snippets import create_position_ids_start_at_padding, DottableDict, modify_variable_mapping
from bert4torch.activations import get_activation
import copy
from packaging import version
from typing import Union, Literal, List


class BERT(PreTrainedModel):
    """构建BERT模型
    """
    _no_split_modules = ["BertLayer"]

    def __init__(
            self,
            vocab_size:int,  # 词表大小
            hidden_size:int,  # 编码维度
            num_hidden_layers:int,  # Transformer总层数
            num_attention_heads:int,  # Attention的头数
            intermediate_size:int,  # FeedForward的隐层维度
            hidden_act:str,  # FeedForward隐层的激活函数
            max_position:int,  # 序列最大长度
            dropout_rate:float=None,  # Dropout比例
            attention_probs_dropout_prob:float=None,  # Attention矩阵的Dropout比例
            embedding_size:int=None,  # 指定embedding_size, 不指定则使用config文件的参数
            keep_hidden_layers:List[int]=None, # 保留的hidden_layer层的id
            residual_attention_scores:bool=False,  # Attention矩阵加残差
            hierarchical_position:Union[bool, float]=None,  # 是否层次分解位置编码
            gradient_checkpoint:bool=False, # 是否使用gradient_checkpoint
            output_all_encoded_layers:bool=False, # 是否返回所有layer的hidden_states
            return_dict:bool=False,  # 是否返回的格式是dict
            tie_word_embeddings:bool=False,  # 是否绑定embedding和lm_head的权重

            segment_vocab_size:int=2,  # segment总数目
            with_pool:bool=False,  # 是否包含Pool部分
            with_nsp:bool=False,  # 是否包含NSP部分
            with_mlm:bool=False,  # 是否包含MLM部分
            # 是否自行传入位置id, True表示传入，False表示不传入，'start_at_padding'表示从padding_idx+1开始
            custom_position_ids:Literal[True, False, 'start_at_padding', 'padding_on_left']=False,
            custom_attention_mask:bool=False, # 是否自行传入attention_mask
            shared_segment_embeddings:bool=False,  # 若True，则segment跟token共用embedding
            conditional_size:Union[bool, int]=None,  # conditional layer_norm
            # additional_embeddng, 是否有额外的embedding, 比如加入词性，音调，word粒度的自定义embedding
            additional_embs:Union[bool, torch.Tensor, List[torch.Tensor]]=False,
            is_dropout:bool=False,
            pad_token_id:int=0,  # 默认0是padding ids, 但是注意google的mt5padding不是0
            layer_type:str='BertLayer',
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        if self.keep_tokens is not None:
            vocab_size = len(self.keep_tokens)
        if self.compound_tokens is not None:
            vocab_size += len(self.compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = kwargs.get('attention_head_size') or self.hidden_size // self.num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)
        self.residual_attention_scores = residual_attention_scores
        self.hierarchical_position = hierarchical_position
        self.gradient_checkpoint = gradient_checkpoint
        self.attention_scores = None
        self.output_all_encoded_layers = output_all_encoded_layers
        self.return_dict = return_dict
        self.tie_word_embeddings = tie_word_embeddings or kwargs.get('tie_emb_prj_weight', False)  # 兼顾old version

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
        self.embeddings = BertEmbeddings(**self.get_kw(*self._embedding_args, **kwargs))
        self.encoderLayer = nn.ModuleList([TRANSFORMER_BLOCKS[layer_type](layer_idx=layer_idx, **self.get_kw(*self._layer_args, **kwargs)) 
                                           if layer_idx in self.keep_hidden_layers else BlockIdentity() for layer_idx in range(self.num_hidden_layers)])
        
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
            # mlm预测
            self.mlmDense = nn.Linear(self.hidden_size, self.embedding_size)  # 允许hidden_size和embedding_size不同
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.embedding_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.tie_weights()
        self.model_type = 'bert'

    @property
    def _embedding_args(self):
        args = ['vocab_size', 'embedding_size', 'hidden_size', 'max_position', 'segment_vocab_size', 
                'shared_segment_embeddings', 'dropout_rate', 'conditional_size']
        return args

    @property
    def _layer_args(self):
        args = ['hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 
                'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'max_position']
        return args
    
    def tie_weights(self):
        """权重的tie"""
        if (self.tie_word_embeddings is True) and self.with_mlm:
            self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmDecoder.bias = self.mlmBias
    
    def get_input_embeddings(self):
        """获取word_embeddings"""
        return self.embeddings.word_embeddings
    
    def layer_forward(self, layer, model_kwargs, use_reentrant=False):
        """transformer block的forward"""
        if self.gradient_checkpoint and self.training:
            if (use_reentrant is True) or version.parse(torch.__version__) < version.parse("1.11.0"):
                # 此种方式要求输入输出是位置参数
                return old_checkpoint(layer, model_kwargs)
            else:
                return checkpoint(layer, use_reentrant=use_reentrant, **model_kwargs)
        else:
            return layer(**model_kwargs)

    def preprare_embeddings_inputs(self, *inputs:Union[tuple, list], **model_kwargs):
        '''解析准备进embedding层的的输入'''
        # ========================= token_ids =========================
        index_ = 0
        if model_kwargs.get('input_ids') is not None:
            token_ids = model_kwargs['input_ids']
        elif model_kwargs.get('token_ids') is not None:
            token_ids = model_kwargs['token_ids']
        else:
            token_ids = inputs[0]
            index_ += 1
        
        # ========================= segment_ids =========================
        if model_kwargs.get('segment_ids') is not None:
            segment_ids = model_kwargs['segment_ids']
        elif model_kwargs.get('token_type_ids') is not None:
            segment_ids = model_kwargs['token_type_ids']
        elif self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None

        # ========================= position_ids =========================
        # 以下只有在一种情况下生效, 是传入了past_key_values接着推理, 如多轮对话中维持past_key_values
        # 1）ptuning_v2不生效(训练阶段), 虽然传入了past_key_values, 但是postion_ids依然从0开始
        # 2）use_states=True推理时候不生效, 虽然past_key_values有值, 但是由于传入了'position_ids'
        past_key_values_length = 0
        if model_kwargs.get('past_key_values_length') is not None:
            past_key_values_length = model_kwargs['past_key_values_length']
        elif model_kwargs.get('past_key_values') is not None:
            past_key_values_length = model_kwargs.get('past_key_values')[0][0].shape[2]
            
        # [btz, seq_len]
        pad_token_id = model_kwargs.get('pad_token_id', self.pad_token_id)
        if model_kwargs.get('position_ids') is not None:
            position_ids = model_kwargs['position_ids']
        elif self.custom_position_ids is True:  # 自定义position_ids
            position_ids = inputs[index_]
            index_ += 1
        elif self.custom_position_ids == 'start_at_padding':
            # 从padding位置开始, 目前使用到的是ethanyt/guwenbert-base和FacebookAI/roberta-base
            position_ids = create_position_ids_start_at_padding(token_ids, pad_token_id, past_key_values_length)
        elif self.custom_position_ids == 'padding_on_left':
            # decoder模型padding在左侧
            position_ids = create_position_ids_start_at_padding(token_ids, pad_token_id, past_key_values_length=-1, start_padding_idx=False)
        else:
            position_ids = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device).unsqueeze(0) + past_key_values_length
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
            attention_mask = (token_ids != pad_token_id).long()  # 默认0为mask_value
            if pad_token_id < 0:
                token_ids = token_ids * attention_mask
        else:  # 自定义word_embedding，目前仅有VAT中使用
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask  # 缓存上次用的attention_mask
        model_kwargs['attention_mask_2d'] = attention_mask
        
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
        
        # attention_mask最后两维是[q_len, k_ken]，如果维度不匹配补齐，目前是在ptuning_v2中使用, 主要为了应对额外传入的past_key_values
        if (model_kwargs.get('past_key_values') is not None) and \
           (attention_mask.shape[-1] < model_kwargs.get('past_key_values')[0][0].shape[2] + token_ids.shape[1]):
            pad_length = model_kwargs.get('past_key_values')[0][0].shape[2] + token_ids.shape[1] - attention_mask.shape[-1]
            pre_attention_mask = torch.ones(attention_mask.shape[:3] + torch.Size([pad_length])).to(attention_mask)
            attention_mask = torch.cat([pre_attention_mask, attention_mask], dim=-1)

        # ========================= conditional layer_norm =========================
        if model_kwargs.get('conditional_emb') is not None:
            conditional_emb = model_kwargs['conditional_emb']
        elif self.conditional_size is not None:
            conditional_emb = inputs[index_]
            index_ += 1
        else:
            conditional_emb = None

        # ========================= additional_embeddng =========================
        # 比如加入词性，音调，word粒度的自定义embedding
        if model_kwargs.get('additional_embs') is not None:
            additional_embs = model_kwargs['additional_embs']
        elif self.additional_embs is True:
            additional_embs = inputs[index_]
            index_ += 1
        else:
            additional_embs = None
        additional_embs = [additional_embs] if isinstance(additional_embs, torch.Tensor) else additional_embs

        # 解析encoder_hidden_state, encoder_attention_mask
        if len(inputs[index_:]) >=2:
            model_kwargs['encoder_hidden_states'], model_kwargs['encoder_attention_mask'] = inputs[index_], inputs[index_+1]
        return token_ids, segment_ids, position_ids, attention_mask, conditional_emb, additional_embs, model_kwargs

    def apply_embeddings(self, *inputs:Union[tuple, list], **model_kwargs):
        """BERT的embedding，可接受"位置参数/关键字参数"形式

        :param inputs: List[torch.Tensor], 默认顺序是[input_ids, segment_ids(若有), position_ids(若有), attention_mask(若有), conditional_emb(若有), additional_embs(若有)]
        :param model_kwargs: Dict[torch.Tensor], 字典输入项，和inputs是二选一的
        :return: Dict[torch.Tensor], [hidden_states, attention_mask, conditional_emb, ...]
        """        
        # 准备进embedding层的一些输入
        input_ids, segment_ids, position_ids, attention_mask, conditional_emb, additional_embs, model_kwargs = \
            self.preprare_embeddings_inputs(*inputs, **model_kwargs)
        
        # 进入embedding层
        hidden_states = self.embeddings(input_ids, segment_ids, position_ids, conditional_emb, additional_embs)
        model_kwargs.update({'hidden_states': hidden_states, 'attention_mask':attention_mask, 'conditional_emb': conditional_emb})
        
        return model_kwargs

    def apply_on_layer_begin(self, l_i, **model_kwargs):
        '''新增对layer block输入进行操作的函数'''
        # if model_kwargs.get('use_states') is not True:
        #     return model_kwargs
        
        if model_kwargs.get('past_key_values') is not None:
            model_kwargs['past_key_value'] = model_kwargs['past_key_values'][l_i]

        if model_kwargs.get('cross_past_key_values') is not None:
            model_kwargs['cross_past_key_value'] = model_kwargs['cross_past_key_values'][l_i]
        return model_kwargs
    
    def apply_on_layer_end(self, l_i, **model_kwargs):
        '''新增对layer block输出进行操作的函数, 目前仅在MixUp中使用'''
        if model_kwargs.get('use_states') is not True:
            return model_kwargs

        if model_kwargs.get('past_key_value') is not None:
            if ('past_key_values' not in model_kwargs) or (model_kwargs.get('past_key_values') is None):
                model_kwargs['past_key_values'] = [None]*self.num_hidden_layers
            model_kwargs['past_key_values'][l_i] = model_kwargs['past_key_value']
        if model_kwargs.get('cross_past_key_value') is not None:
            if ('cross_past_key_values' not in model_kwargs) or (model_kwargs.get('cross_past_key_values') is None):
                model_kwargs['cross_past_key_values'] = [None]*self.num_hidden_layers
            model_kwargs['cross_past_key_values'][l_i] = model_kwargs['cross_past_key_value']
        return model_kwargs
    
    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        
        :param model_kwargs: Dict[torch.Tensor], 包含hidden_states, attention_mask, conditional_emb等
        :return: Dict[torch.Tensor], [encoded_layers, conditional_emb]
        """
        encoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.encoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
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

        :param model_kwargs: Dict[torch.Tensor], 包含encoded_layers, conditional_emb等
        :return: List[torch.Tensor]|torch.Tensor|Dict[torch.Tensor], 模型输出，包含last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)
        """
        # 获取最后一层隐藏层的输出
        encoded_layers, conditional_emb = model_kwargs['encoded_layers'], model_kwargs.get('conditional_emb', None)
        last_hidden_state = encoded_layers[-1]

        # 是否添加pool层
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(last_hidden_state[:, 0]))
        else:
            pooled_output = None
        # 是否添加nsp
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        # 是否添加mlm
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(last_hidden_state)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state, conditional_emb)
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation('linear' if self.with_mlm is True else self.with_mlm)
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None

        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            return self.gen_outputs(locals(), last_hidden_state, pooled_output, mlm_scores, nsp_scores)
        else:
            return self.gen_outputs(locals(), encoded_layers, pooled_output, mlm_scores, nsp_scores)        
        
    def gen_outputs(self, locals_dict, *args):
        ''' 生成outputs list/dict两种形式'''
        if not self.return_dict:
            # 不以dict格式返回
            outputs = [value for value in args if value is not None]
            return outputs if len(outputs) > 1 else outputs[0]
        else:
            # 以dict格式返回
            outputs = DottableDict()
            for arg in args:
                if arg is None:
                    continue
                # 获取变量名
                for name, value in locals_dict.items():
                    if value is arg:
                        outputs[name] = arg
                        break
            return outputs

    def load_trans_ckpt(self, checkpoint):
        """加载ckpt, 方便后续继承并做一些预处理
        这么写的原因是下游很多模型从BERT继承，这样下游可以默认使用PreTrainedModel的load_trans_ckpt
        """
        state_dict = super().load_trans_ckpt(checkpoint)        
        if hasattr(self, 'model_type') and (self.model_type == 'bert'):
            # bert
            mapping_reverse = {v:k for k, v in self.variable_mapping().items()}
            mapping = {}
            for key in state_dict.keys():
                # bert-base-chinese中ln的weight和bias是gamma和beta
                if ".gamma" in key:
                    value = key.replace(".gamma", ".weight")
                    mapping[mapping_reverse[value]] = key
                if ".beta" in key:
                    value = key.replace(".beta", ".bias")
                    mapping[mapping_reverse[value]] = key
            if ('cls.predictions.bias' in state_dict) and ('cls.predictions.decoder.bias' not in state_dict):
                mapping['mlmDecoder.bias'] = 'cls.predictions.bias'
            self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict
    
    def load_variable(self, variable, ckpt_key, model_key, prefix='bert'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        # mapping = self.variable_mapping()

        if ckpt_key in {
            f'{prefix}.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif model_key in {'embeddings.word_embeddings.weight', 'mlmBias',
                           'mlmDecoder.weight', 'mlmDecoder.bias'}:
            # bert4torch中model_key相对固定, 能cover住绝大多数BERT子类
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        """权重映射字典，格式为{model_key: ckpt_key}"""
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
                            f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'attention.output.LayerNorm.weight',
                            f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'attention.output.LayerNorm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias',
                            f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'output.LayerNorm.weight',
                            f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'output.LayerNorm.bias'
                            })

        if self.embedding_size != self.hidden_size:
            mapping.update({'embeddings.embedding_hidden_mapping_in.weight': f'bert.encoder.embedding_hidden_mapping_in.weight',
                            'embeddings.embedding_hidden_mapping_in.bias': f'bert.encoder.embedding_hidden_mapping_in.bias'})
        return mapping
