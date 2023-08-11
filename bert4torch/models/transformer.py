from bert4torch.models.bert import BERT
from bert4torch.models.base import LM_Mask, BERT_BASE
from bert4torch.snippets import delete_arguments, insert_arguments
from bert4torch.activations import get_activation
from bert4torch.layers import LayerNorm
from torch import nn
import torch


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
    @insert_arguments(with_lm=True)
    def __init__(self, *args, logit_scale=False, final_layernorm=False, **kwargs):
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True  # 标记是decoder
        super().__init__(*args, **kwargs)
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.final_layernorm = final_layernorm

        # 从hidden_states映射到logit
        if self.with_lm:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.final_activation = get_activation('linear' if self.with_lm is True else self.with_lm)  # 添加激活，一般是线性激活或softmax

            # decoder底层的embedding和顶层的全连接共享
            # [True]: fudan_bart和uer_t5的t5, [False]: mt5和t5_pegasus
            self.tie_weights()
            if logit_scale:  # T5默认会有logit_scale, bart默认没有
                self.logit_scale = (self.hidden_size ** -0.5)
        
        if self.final_layernorm:
            self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12), 
                                            conditional_size=self.conditional_size, norm_mode=kwargs.get('norm_mode', 'normal'),
                                            weight=kwargs.get('weight', True), bias=kwargs.get('bias', True))

    def tie_weights(self):
        if self.tie_emb_prj_weight is True:
            self.lm_head.weight = self.embeddings.word_embeddings.weight

    def apply_main_layers(self, **model_kwargs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块；
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        decoded_layers = [model_kwargs['hidden_states']] # 添加embedding的输出
        for l_i, layer_module in enumerate(self.decoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
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
        hidden_states = model_kwargs['decoded_layers'][-1]  # decoder顶层的hidden_states [btz, seq_len, hdsz]

        if self.final_layernorm:
            hidden_states = self.LayerNormFinal(hidden_states)
        
        if self.with_lm:
            logits = self.lm_head(hidden_states)  # [btz, seq_len, vocab_size]
            logits = logits * self.logit_scale if hasattr(self, 'logit_scale') else logits
            logits = self.final_activation(logits)
            return logits
        return hidden_states

    def load_variable(self, state_dict, name, prefix='bert'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        variable = state_dict[name]
        if name in {f'{prefix}.embeddings.word_embeddings.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        elif name == f'{prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        else:
            return variable
        
    def variable_mapping(self, prefix='bert'):
        raw_mapping = super().variable_mapping(prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        
        if self.final_layernorm:
            mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight',
                            'LayerNormFinal.bias': f'{prefix}.LayerNormFinal.bias'})
        if self.with_lm and (not self.tie_emb_prj_weight):  # 当且仅当未绑定权重的时候
            mapping.update({'lm_head.weight': f'{prefix}.lm_head.weight'})
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

    def tie_weights(self):
        self.decoder.tie_weights()

    def forward(self, *inputs):
        inputs = self.args_segmentate(inputs)
        encoder_input, decoder_input = inputs[:2]

        # encoder
        encoder_hidden_states, encoder_attention_mask = self.encoder(encoder_input)

        # decoder
        decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])
        
        # 输出encoder_hidden_state和decoder_hidden_state，以应对一些多任务情况
        # with_lm=True时候，decoder_outputs为logits, False时候为decoder_hidden_state
        return [encoder_hidden_states] + [decoder_outputs]

