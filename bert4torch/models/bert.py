import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from bert4torch.models.base import BERT_BASE
from bert4torch.layers import LayerNorm, BertEmbeddings
from bert4torch.layers import LayerNorm, BertEmbeddings, BertLayer, BlockIdentity
from bert4torch.snippets import old_checkpoint, create_position_ids_start_at_padding, DottableDict
from bert4torch.activations import get_activation
import copy
from packaging import version


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
        self.prefix = 'bert'
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
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.tie_weights()
        # 下述继承于BERT的有声明新的参数，在这里初始化不能统一初始化到

    def tie_weights(self):
        """权重的tie"""
        if self.tie_emb_prj_weight is True:
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

    def apply_embeddings(self, *inputs, **model_kwargs):
        """BERT的embedding是token、position、segment三者embedding之和

        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有), additional_input(若有)]
        :return: List[torch.Tensor], [hidden_states, attention_mask, conditional_emb, ...]
        """
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'

        # ========================= token_ids =========================
        if model_kwargs.get('input_ids') is not None:
            token_ids = model_kwargs['input_ids']
        elif model_kwargs.get('token_ids') is not None:
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

        :param inputs: List[torch.Tensor], 默认顺序为[encoded_layers, conditional_emb]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
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
        """加载ckpt, 方便后续继承并做一些预处理"""
        state_dict = torch.load(checkpoint, map_location='cpu')
        old_new_keys = {}
        for key in state_dict.keys():
            # bert-base-chinese中ln的weight和bias是gamma和beta
            if ".gamma" in key:
                old_new_keys[key] = key.replace(".gamma", ".weight")
            if ".beta" in key:
                old_new_keys[key] = key.replace(".beta", ".bias")
        for old_key, new_key in old_new_keys.items():
            state_dict[new_key] = state_dict.pop(old_key)
        if ('cls.predictions.bias' in state_dict) and ('cls.predictions.decoder.bias' not in state_dict):
            state_dict['cls.predictions.decoder.bias'] = state_dict['cls.predictions.bias']
        return state_dict    

    def load_variable(self, state_dict, name):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        variable = state_dict[name]
        if name in {
            f'{self.prefix}.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == f'{self.prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'cls.seq_relationship.weight':
            return variable.T
        else:
            return variable

    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': f'{self.prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': f'{self.prefix}.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': f'{self.prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': f'{self.prefix}.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': f'{self.prefix}.embeddings.LayerNorm.bias',
            'pooler.weight': f'{self.prefix}.pooler.dense.weight',
            'pooler.bias': f'{self.prefix}.pooler.dense.bias',
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
            prefix_i = f'{self.prefix}.encoder.layer.%d.' % i
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
            mapping.update({'embeddings.embedding_hidden_mapping_in.weight': f'{self.prefix}.encoder.embedding_hidden_mapping_in.weight',
                            'embeddings.embedding_hidden_mapping_in.bias': f'{self.prefix}.encoder.embedding_hidden_mapping_in.bias'})
        return mapping
