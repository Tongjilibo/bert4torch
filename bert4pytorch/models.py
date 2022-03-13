import torch
import torch.nn as nn
import copy
import json
from itertools import cycle
from bert4pytorch.layers import LayerNorm, activations, BertEmbeddings, BertLayer, Identity
from bert4pytorch.snippets import ProgbarLogger, metric_mapping, search_layer

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()
    
    def compile(self, loss, optimizer, scheduler=None, metrics=None, adversarial_train={'name': ''}):
        '''定义loss，optimizer，metrics，是否在计算loss前reshape
        loss: 
        optimizer:
        metrics:
        '''
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        if metrics is None:
            metrics = []
        self.metrics = ['loss'] + metrics
        self.adversarial = adversarial_train


    def fit(self, train_dataloader, steps_per_epoch=None, epochs=1, callbacks=[]):
        steps_per_epoch = len(train_dataloader) if steps_per_epoch is None else steps_per_epoch

        train_dataloader = cycle(train_dataloader)
        callbacks = [ProgbarLogger(epochs, steps_per_epoch, self.metrics)] + callbacks

        for callback in callbacks:
            callback.on_train_begin()  #callback

        global_step = 0
        # 对抗训练
        if self.adversarial['name'] == 'fgm':
            ad_train = FGM(self)
        elif self.adversarial['name'] == 'pgd':
            ad_train = PGD(self)
        elif self.adversarial['name'] == 'gradient_penalty':
            pass
        elif self.adversarial['name'] == '':  # 默认情况
            pass
        else:
            raise ValueError('adversarial_train only support fgm and pgd mode')
        self.adversarial['K'] = self.adversarial.get('K', 3)
        self.adversarial['epsilon'] = self.adversarial.get('epsilon', 1.0)
        self.adversarial['emb_name'] = self.adversarial.get('emb_name', 'emb')
        self.adversarial['alpha'] = self.adversarial.get('alpha', 0.3)

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(global_step, epoch, {})  # callback
            for bti in range(steps_per_epoch):
                batch = next(train_dataloader)
                train_X, train_y = batch

                if isinstance(train_X, (list, tuple)):  # 仅允许嵌套两层，即((token_ids1, mask1), (token_ids2, mask2))
                    if isinstance(train_X[0], (list, tuple)):
                        btz = train_X[0][0].size(0)
                    else:
                        btz = train_X[0].size(0)
                elif isinstance(train_X, torch.Tensor):
                    btz = train_X.size(0)
                else:
                    raise ValueError('Input only support [list, tuple, tensor]')
                logs = {'batch': bti, 'size': btz}
                for callback in callbacks:
                    callback.on_batch_begin(global_step, bti, logs)  # callback

                self.train()  # 设置为train模式
                # 入参个数判断，如果入参>=3表示是多个入参，如果=2则表示是一个入参
                output = self.forward(*train_X) if self.forward.__code__.co_argcount >= 3 else self.forward(train_X)
                loss = self.criterion(output, train_y)
                loss.backward(retain_graph=(self.adversarial['name'] == 'gradient_penalty'))
                
                # 对抗训练
                if self.adversarial['name'] == 'fgm':
                    ad_train.attack(epsilon=self.adversarial['epsilon'], emb_name=self.adversarial['emb_name']) # embedding被修改了
                    output = self.forward(*train_X) if self.forward.__code__.co_argcount >= 3 else self.forward(train_X)
                    loss = self.criterion(output, train_y)
                    loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                    ad_train.restore(emb_name=self.adversarial['emb_name']) # 恢复Embedding的参数
                elif self.adversarial['name'] == 'pgd':
                    for t in range(self.adversarial['K']):
                        ad_train.attack(epsilon=self.adversarial['epsilon'], alpha=self.adversarial['alpha'], 
                                        emb_name=self.adversarial['emb_name'], is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != self.adversarial['K']-1:
                            self.optimizer.zero_grad()
                        else:
                            ad_train.restore_grad() # 恢复正常的grad
                        output = self.forward(*train_X) if self.forward.__code__.co_argcount >= 3 else self.forward(train_X)
                        loss = self.criterion(output, train_y)
                        loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                    ad_train.restore(emb_name=self.adversarial['emb_name']) # 恢复embedding参数
                # 梯度惩罚
                elif self.adversarial['name'] == 'gradient_penalty':
                    para = search_layer(self, self.adversarial['emb_name'], retrun_first=True)
                    gp = (para.grad ** 2).sum()
                    loss += 0.5 * gp * self.adversarial['epsilon']
                    loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                logs.update({'loss': loss.item()})
                for metric in self.metrics[1:]:
                    logs[metric] = metric_mapping(metric, output, train_y)
                for callback in callbacks:
                    callback.on_batch_end(global_step, bti, logs)  #callback

                global_step += 1
            for callback in callbacks:
                callback.on_epoch_end(global_step, epoch, logs)  #callback
    
    def predict(self, input_tensor_list, return_all=None):
        self.eval()
        with torch.no_grad():
            if self.forward.__code__.co_argcount >= 3:
                output = self.forward(*input_tensor_list)
            else:
                output = self.forward(input_tensor_list)
        if return_all is None:
            return output
        elif isinstance(output, (tuple, list)) and isinstance(return_all, int) and return_all < len(output):
            return output[return_all]
        else:
            raise ValueError('Return format error')
    
    def load_weights(self, load_path, strict=True):
        state_dict = torch.load(load_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=strict)

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)


class BERT_BASE(BaseModel):
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
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            ignore_invalid_weights=False,  # 允许跳过不存在的权重
            keep_hidden_layers=None, # 保留的hidden_layer层的id
            **kwargs
    ):
        super(BERT_BASE, self).__init__(**kwargs)
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
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches: 为Attention的K,V的缓存序列字典，格式为{Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数: 实现Conditional Layer Normalization时使用，用来实现以“固定长度向量”为条件的条件Bert。
        """
        # additional_input
        # if additional_input_layers is not None:
        #     if not isinstance(additional_input_layers, list):
        #         self.additional_input_layers = [additional_input_layers]
        #     else:
        #         self.additional_input_layers = additional_input_layers

        # Other
        self.attention_caches = attention_caches or {}
        # self.layer_norm_conds = [
        #     layer_norm_cond,
        #     layer_norm_cond_hidden_size,
        #     layer_norm_cond_hidden_act or 'linear',
        # ]
        self.output_all_encoded_layers = True if 'output_all_encoded_layers' in kwargs else False

    def forward(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        outputs = self.apply_main_layers(outputs)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def init_model_weights(self, module):
        raise NotImplementedError

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_embeddings(self, embeddings):
        """根据keep_tokens和compound_tokens对embedding进行修改
        """
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                ext_embeddings.append(
                    torch.mean(embeddings[item], 0) * torch.ones_like(embeddings[item])
                )
            embeddings = torch.concat([embeddings]+ext_embeddings, 0)

        return embeddings

    def load_pos_embeddings(self, embeddings):
        """根据hierarchical_position对pos_embedding进行修改
        """
        if self.hierarchical_position is not None:
            alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]
            embeddings_x = torch.take_along_dim(embeddings,  torch.div(position_index, embeddings.size(0), rounding_mode='trunc'), dim=0)
            embeddings_y = torch.take_along_dim(embeddings, position_index % embeddings.size(0), dim=0)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y

        return embeddings

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        # 加载模型文件
        state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        state_dict_new ={}
        for new_key, old_key in mapping.items():
            if old_key in state_dict:
                state_dict_new[new_key] = self.load_variable(state_dict, old_key)
            else:
                print(f'[WARNIMG] {old_key} not found in the model file')
        del state_dict

        # 将ckpt的权重load到模型结构中
        self.load_state_dict(state_dict_new, strict=self.ignore_invalid_weights)

    
    # def get_inputs(self):
    #     pass
    
    # def set_inputs(self, inputs, additional_input_layers=None):
    #     """设置input和inputs属性
    #     """
    #     pass

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

def lm_mask(segment_ids):
    """定义下三角Attention Mask（语言模型用）
    """
    idxs = torch.arange(0, segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask


def unilm_mask(segment_ids):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
        其中source和target的分区，由segment_ids来表示。
        UniLM: https://arxiv.org/abs/1905.03197
    """

    # 在序列维度进行累加求和
    idxs = torch.cumsum(segment_ids, dim=1)
    # 构造unilm的mask矩阵，并把shape扩充到[batch_size, num_heads, from_seq_length, to_seq_length]
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).to(dtype=torch.float32)
    return mask


####################################################################################
#       bert                                                                       #
####################################################################################

class BERT(BERT_BASE):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            initializer_range=0.02, # 权重初始化方差
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            hierarchical_position=None,  # 是否层次分解位置编码
            custom_position_ids=False,  # 是否自行传入位置id
            custom_attention_mask=False, # 是否自行传入attention_mask
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            layer_norm_cond=None,  # conditional layer_norm
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.initializer_range = initializer_range
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.layer_norm_conds = layer_norm_cond
        self.conditional_size = layer_norm_cond.weight.size(1) if layer_norm_cond is not None else None
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate, self.conditional_size)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh()
            if self.with_nsp:
                # Next Sentence Prediction部分
                # nsp的输入为pooled_output, 所以with_pool为True是使用nsp的前提条件
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            # self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = activations[self.hidden_act]
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size)
        self.apply(self.init_model_weights)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        默认第一个是token_ids, 第二个是segment_ids, 第三个是position_ids
        """
        token_ids = inputs[0]
        index_ = 1
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None

        if self.custom_position_ids:
            position_ids = inputs[index_]
            index_ += 1
        else:
            position_ids = None
        # 根据token_ids创建一个3D的attention mask矩阵，尺寸为[batch_size, 1, 1, to_seq_length]，
        # 目的是为了适配多头注意力机制，从而能广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
        if self.custom_attention_mask:
            attention_mask = inputs[index_]
            index_ += 1
        else:
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)  # 这里指定了0为mask_value
        
        self.compute_attention_bias([token_ids, segment_ids])  # 根据lm或者unilm需要对mask做调整
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # 兼容fp16
        
        # 对mask矩阵中，数值为0的转换成很大的负数，使得不需要attention的位置经过softmax后,分数趋近于0
        # attention_mask = (1.0 - attention_mask) * -10000.0
        # 执行embedding
        if self.layer_norm_conds is None:
            conditional_emb = None
        else:
            conditional_emb = self.layer_norm_conds(inputs[index_])
        hidden_states = self.embeddings(token_ids, segment_ids, conditional_emb)
        if len(inputs[index_:]) > 0:
            return hidden_states, attention_mask, conditional_emb, inputs[index_:]
        else:
            return hidden_states, attention_mask, conditional_emb

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        默认第一个是hidden_states, 第二个是attention_mask, 第三个是conditional_emb
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        for layer_module in self.encoderLayer:
            hidden_states = layer_module(hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]
    
    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        # 获取最后一层隐藏层的输出
        encoded_layers, conditional_emb = inputs
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
        else:
            mlm_scores = None
        
        outputs = [value for value in [encoded_layers, pooled_output, mlm_scores, nsp_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]

    def load_variable(self, state_dict, name):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {
            'bert.embeddings.word_embeddings.weight',
            'cls.predictions.bias',
            'cls.predictions.decoder.weight',
            'cls.predictions.decoder.bias'
        }:
            return self.load_embeddings(variable)
        elif name == 'bert.embeddings.position_embeddings.weight':
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

        return mapping


class ALBERT(BERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([layer])

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        for _ in range(self.num_hidden_layers):
            hidden_states = self.encoderLayer[0](hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

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
        """加载单个变量的函数
        """
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
        super(ALBERT_Unshared).__init__(*args, **kwargs)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states] # 添加embedding的输出
        for i in range(self.num_hidden_layers):
            hidden_states = self.encoderLayer[i](hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]


class NEZHA(BERT):
    """华为推出的NAZHA模型
    链接：https://arxiv.org/abs/1909.00204
    """
    def __init__(self, *args, **kwargs):
        super(NEZHA, self).__init__(*args, **kwargs)
        # 通过max_position=0控制在embedding阶段无位置编码
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, 0, self.segment_vocab_size, self.dropout_rate, self.conditional_size)
        config = {'p_bias': 'typical_relative', 'max_position_embeddings': self.max_position, 'max_relative_position': kwargs.get('max_relative_position')}
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, 
                            conditional_size=self.conditional_size, **config)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

class RoFormer(BERT):
    """旋转式位置编码的BERT模型
    链接：https://kexue.fm/archives/8265
    """
    def __init__(self, *args, **kwargs):
        super(RoFormer, self).__init__(*args, **kwargs)
        # 通过max_position=0控制在embedding阶段无位置编码
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, 0, self.segment_vocab_size, self.dropout_rate, self.conditional_size)
        config = {'p_bias': 'rotary', 'max_position_embeddings': self.max_position}
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, 
                            conditional_size=self.conditional_size, **config)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])

class Transformer(BERT):
    '''encoder-decoder结构
    '''
    def __init__(self, *args, emb_src_tgt_weight_sharing=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self.src_vocab_size = kwargs.get('src_vocab_size') or self.vocab_size
        self.embeddings = BertEmbeddings(self.src_vocab_size, self.embedding_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate, self.conditional_size)
        # decoder
        self.tgt_vocab_size = kwargs.get('tgt_vocab_size') or self.vocab_size
        if emb_src_tgt_weight_sharing:
            # encoder和decoder的embedding权重共享
            assert self.src_vocab_size == self.tgt_vocab_size, "To share word embedding, the vocab size of src/tgt shall be the same."
            self.tgt_embeddings = self.embeddings
        else:
            self.tgt_embeddings = BertEmbeddings(self.tgt_vocab_size, self.embedding_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate, self.conditional_size)
        dec_layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size, is_decoder=True)
        self.decoderLayer = nn.ModuleList([copy.deepcopy(dec_layer) if layer_id in self.keep_hidden_layers else Identity() for layer_id in range(self.num_hidden_layers)])


    def forward(self, inputs):
        """定义模型的执行流程
        """
        encoder_input, decoder_input = inputs[:2]

        # encoder
        encoder_emb = self.apply_embeddings(encoder_input)
        encode_outputs, _ = self.apply_main_layers(encoder_emb)
        
        # decoder
        decoder_emb = self.apply_tgt_embeddings(decoder_input)
        encoder_hidden_state = encode_outputs[-1]
        encoder_attention_mask = encoder_emb[1]
        decoder_outputs, _ = self.apply_tgt_main_layers([*decoder_emb, encoder_hidden_state, encoder_attention_mask])
        decoder_hidden_state = decoder_outputs[-1]

        return decoder_hidden_state
    
    def apply_tgt_embeddings(self, inputs):
            """tgt的embedding是token、position、segment三者embedding之和
            """
            token_ids = inputs[0]
            index_ = 1
            if self.segment_vocab_size > 0:
                segment_ids = inputs[index_]
                index_ += 1
            else:
                segment_ids = None

            if self.custom_position_ids:
                position_ids = inputs[index_]
                index_ += 1
            else:
                position_ids = None
            # 根据token_ids创建一个3D的attention mask矩阵，尺寸为[batch_size, 1, 1, to_seq_length]，
            # 目的是为了适配多头注意力机制，从而能广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
            if self.custom_attention_mask:
                attention_mask = inputs[index_]
                index_ += 1
            else:
                attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)  # 这里指定了0为mask_value
            
            # 加三角式mask [btz, 1, seq_q, seq_k]
            attention_mask = (attention_mask + self.get_subsequent_mask(attention_mask)).gt(1).to(attention_mask)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # 兼容fp16
            
            # 对mask矩阵中，数值为0的转换成很大的负数，使得不需要attention的位置经过softmax后,分数趋近于0
            # attention_mask = (1.0 - attention_mask) * -10000.0
            # 执行embedding
            if self.layer_norm_conds is None:
                conditional_emb = None
            else:
                conditional_emb = self.layer_norm_conds(inputs[index_])
            hidden_states = self.tgt_embeddings(token_ids, segment_ids, conditional_emb)
            return hidden_states, attention_mask, conditional_emb

    def apply_tgt_main_layers(self, inputs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask = inputs[:5]
        decoded_layers = [hidden_states] # 添加embedding的输出
        for i, layer_module in enumerate(self.decoderLayer):
            hidden_states = layer_module(hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask)
            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        return [decoded_layers, conditional_emb]

    @staticmethod
    def get_subsequent_mask(seq):
        ''' 下三角为1表示可以访问'''
        sz_b, _, _, len_s = seq.size()
        subsequent_mask = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1).expand(sz_b, -1, -1, -1)  # b x 1 x ls x ls
        return subsequent_mask

class BART(Transformer):
    '''encoder-decoder结构
    '''
    def __init__(self, *args, emb_src_tgt_weight_sharing=False, **kwargs):
        super(BART, self).__init__(*args, emb_src_tgt_weight_sharing=False, **kwargs)
        self.emb_src_tgt_weight_sharing = emb_src_tgt_weight_sharing

    def load_variable(self, state_dict, name):
        """加载单个变量的函数
        """
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
            'embeddings.word_embeddings.weight': 'shared.weight' if self.emb_src_tgt_weight_sharing else 'encoder.embed_tokens.weight',
            'embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
            'embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight',
            'embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias',
            'tgt_embeddings.word_embeddings.weight': 'shared.weight' if self.emb_src_tgt_weight_sharing else 'decoder.embed_tokens.weight',
            'tgt_embeddings.position_embeddings.weight': 'encoder.embed_positions.weight',
            'tgt_embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight',
            'tgt_embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias',

        }
        for i in range(self.num_hidden_layers):
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight',
                            f'encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias',
                            f'encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight',
                            f'encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias',
                            f'encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight',
                            f'encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias',
                            f'encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight',
                            f'encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias',
                            f'encoderLayer.{i}.layerNorm1.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm1.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias',
                            f'encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight',
                            f'encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias',
                            f'encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight',
                            f'encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias',
                            f'encoderLayer.{i}.layerNorm2.weight': f'encoder.layers.{i}.final_layer_norm.weight',
                            f'encoderLayer.{i}.layerNorm2.bias': f'encoder.layers.{i}.final_layer_norm.bias',
                            f'decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight',
                            f'decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias',
                            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight',
                            f'decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias',
                            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight',
                            f'decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias',
                            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight',
                            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias',
                            f'decoderLayer.{i}.layerNorm1.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight',
                            f'decoderLayer.{i}.layerNorm1.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias',
                            f'decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight',
                            f'decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias',
                            f'decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight',
                            f'decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias',
                            f'decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight',
                            f'decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias',
                            f'decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight',
                            f'decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias',
                            f'decoderLayer.{i}.layerNorm3.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight',
                            f'decoderLayer.{i}.layerNorm3.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias',
                            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight',
                            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias',
                            f'decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight',
                            f'decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias',
                            f'decoderLayer.{i}.layerNorm2.weight': f'decoder.layers.{i}.final_layer_norm.weight',
                            f'decoderLayer.{i}.layerNorm2.bias': f'decoder.layers.{i}.final_layer_norm.bias'

                            })

        return mapping

def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        return_model_config=False,
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    
    # bart单独处理下
    if isinstance(model, str) and model=='bart':
        configs['hidden_size'] = configs.get('d_model') if 'hidden_size' not in configs else configs['hidden_size']
        # 这里也可以用 decoder_attention_heads
        configs['num_attention_heads'] = configs.get('encoder_attention_heads') if 'num_attention_heads' not in configs else configs['num_attention_heads']
        # 这里也可以用 decoder_ffn_dim
        configs['intermediate_size'] = configs.get('encoder_ffn_dim') if 'intermediate_size' not in configs else configs['intermediate_size']
        configs['hidden_act'] = configs.get('activation_function', 'gelu')
        configs['max_position'] = configs.get('max_position_embeddings', 512) + 2
        configs['dropout_rate'] = configs.get('dropout')
        configs['attention_probs_dropout_prob'] = configs.get('attention_dropout')

    models = {
        'bert': BERT,
        'roberta': BERT,  
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'nezha': NEZHA,
        'roformer': RoFormer,
        'transformer': Transformer,
        'bart': BART
    }

    if isinstance(model, str):  # string表示使用自带的模型
        MODEL = models[model]
    elif isinstance(model, type) and issubclass(model, BERT_BASE): # nn.Module表示使用自定义的模型：
        MODEL = model
    else:
        raise ValueError('"model" args type should be string or nn.Module')

    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5', ]:
        raise ValueError(f'"{model}" model can not be used as "{application}" application.\n')

    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    if return_model_config:
        return transformer, configs
    else:
        return transformer


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        seq_len = inputs[0].shape[1]
        attention_bias = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device), diagonal=0)
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias

def extend_with_language_model(InputModel):
    """添加下三角的Attention Mask（语言模型用）
    """
    class LanguageModel(LM_Mask, InputModel):
        """带下三角Attention Mask的派生模型
        """
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(with_mlm=True, *args, **kwargs)

    return LanguageModel

class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        segment_ids = inputs[1]
        attention_bias = torch.cumsum(segment_ids, dim=1)
        attention_bias = (attention_bias.unsqueeze(1)) <= (attention_bias.unsqueeze(2))
        self.attention_bias = attention_bias.unsqueeze(1).long()

        return self.attention_bias

def extend_with_unified_language_model(InputModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）
    """
    class UnifiedLanguageModel(UniLM_Mask, InputModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(with_mlm=True, *args, **kwargs)

    return UnifiedLanguageModel

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self, epsilon=1., alpha=0.3, emb_name='emb', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]