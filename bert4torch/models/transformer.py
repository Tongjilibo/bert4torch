from bert4torch.models.bert import BERT
from bert4torch.models.base import LM_Mask, PreTrainedModel
from bert4torch.snippets import delete_arguments, insert_arguments
from bert4torch.activations import get_activation
from bert4torch.layers import LayerNorm
from bert4torch.generation import SeqGeneration, Seq2SeqGeneration
from typing import Union, Literal
from torch import nn
import torch


class Encoder(BERT):
    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = kwargs.get('src_vocab_size', kwargs['vocab_size'])
        super().__init__(*args, **kwargs)
        # encoder需要返回encoder_attention_mask
        self.encoder_attention_mask = None
        self.model_type = 'encoder'
    
    def forward(self, *inputs, **model_kwargs):
        """因为encoder需要返回encoder_attention_mask，因此这里从新定义一下，多返回一个参数
        """
        # 返回model_kwargs方便解析attention_mask
        outputs, model_kwargs = super().forward(*inputs, use_states=True, **model_kwargs)
        # return: [encoder_hidden_states, encoder_attention_mask]
        return ([outputs] if isinstance(outputs, torch.Tensor) else outputs) + [model_kwargs['attention_mask']]


class PreTrainedModelForDecoder(PreTrainedModel):
    passed_kwargs = {'use_states', 'position_ids', 'past_token_ids', 'attention_mask_2d', 
                     'attention_mask', 'past_key_values', 'cross_past_key_values'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def prepare_inputs_for_generation(self, *inputs, **states):
        '''每次generate时候在self.forward前准备所需参数，方便继承'''
        return states

    def _get_initial_model_kwargs(self, model_kwargs:dict):
        '''初始化的时候去掉不需要要素'''
        # self.num_logits_to_keep = 1  # generate的时候,只计算最后一个logit
        if model_kwargs.get('states') is not None:
            return model_kwargs['states']
        states = {k:v for k,v in model_kwargs.items() if k in self.passed_kwargs}
        return states if states else None
    
    def _update_model_kwargs_for_generation(self, outputs:Union[torch.Tensor, list, tuple, dict], model_kwargs:dict):
        '''需要返回给下一次generate使用到的要素，方便继承'''
        if model_kwargs.get('states') is not None:
            return model_kwargs['states']
        states = {k:v for k,v in model_kwargs.items() if k in self.passed_kwargs}
        return states if states else None

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
        else:
            return outputs

    def _prepare_generation(self, **kwargs):
        if not hasattr(self, 'generation'):
            self.generation = SeqGeneration(self, **kwargs)

    def generate(self, input_ids:Union[str, list, torch.Tensor], **kwargs):
        '''单条样本生成 / batch样本生成，use_states=True时要求padding_side='left'
        '''
        self._prepare_generation(**kwargs)
        return self.generation.generate(input_ids, **kwargs)

    def stream_generate(self, input_ids:Union[str, torch.Tensor], **kwargs):
        '''单条样本stream输出预测的结果'''
        self._prepare_generation(**kwargs)
        yield from self.generation.stream_generate(input_ids, **kwargs)


class Decoder(LM_Mask, BERT, PreTrainedModelForDecoder):
    '''所有decoder模型的基类(含大模型)'''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    @insert_arguments(with_lm=True)
    def __init__(self, *args, logit_scale:bool=False, final_layernorm:bool=False, 
                 convert_logits_dtype:Literal['float16', 'float32', 'float64', 'bfloat16', None]=None, **kwargs):
        '''
        :param logit_scale: bool, 是否对logits进行缩放
        :param final_layernorm: bool, 对last_hidden_state是否进行层归一化
        :param convert_logits_dtype: bool, 是否对logits进行dtype转换
        '''
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True  # 标记是decoder
        super().__init__(*args, **kwargs)
        self.is_decoder = True
        self.model_type = 'decoder'
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.final_layernorm = final_layernorm
        mapping = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32, 'float64': torch.float64}
        self.convert_logits_dtype = mapping[convert_logits_dtype] if convert_logits_dtype is not None else None
        self.num_logits_to_keep = kwargs.get('num_logits_to_keep', 0)

        # 从hidden_states映射到logit
        if self.with_lm:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.final_activation = get_activation('linear' if self.with_lm is True else self.with_lm)  # 添加激活，一般是线性激活或softmax
        self.tie_weights()

        if isinstance(logit_scale, bool) and logit_scale:
            # bool类型，T5默认会有logit_scale, bart默认没有
            self.logit_scale = (self.hidden_size ** -0.5)
        elif not isinstance(logit_scale, bool) and isinstance(logit_scale, (int, float)):
            self.logit_scale = logit_scale
        
        if self.final_layernorm:
            self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12), 
                                            conditional_size=self.conditional_size, norm_mode=kwargs.get('norm_mode', 'normal'), 
                                            rmsnorm_fp32=kwargs.get('rmsnorm_fp32', 'llama-qwen'),
                                            weight=kwargs.get('weight', True), bias=kwargs.get('bias', True))

    def tie_weights(self):
        # decoder底层的embedding和顶层的全连接共享
        # [True]: fudan_bart和uer_t5的t5, [False]: mt5和t5_pegasus
        if self.tie_word_embeddings and self.with_lm:
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
        last_hidden_state = model_kwargs['decoded_layers'][-1]  # decoder顶层的hidden_states [btz, seq_len, hdsz]

        if self.final_layernorm:
            last_hidden_state = self.LayerNormFinal(last_hidden_state)
        
        if self.with_lm:
            logits = self.lm_head(last_hidden_state[:, -self.num_logits_to_keep:, :])  # [btz, seq_len, vocab_size]
            logits = logits * self.logit_scale if hasattr(self, 'logit_scale') else logits
            logits = self.final_activation(logits)
            if self.convert_logits_dtype is not None:
                logits = logits.to(self.convert_logits_dtype)
            return self.gen_outputs(locals(), last_hidden_state, logits) if self.return_dict else logits
        elif not self.return_dict:
            return last_hidden_state
        else:
            return self.gen_outputs(locals(), last_hidden_state)

    def load_variable(self, variable, ckpt_key, model_key, prefix='decoder'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        # mapping = self.variable_mapping()
        # if ckpt_key in {f'{prefix}.embeddings.word_embeddings.weight', f'{prefix}.lm_head.weight'}:
        #     return self.load_embeddings(variable)
        if model_key in {'embeddings.word_embeddings.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable
        
    def variable_mapping(self, prefix='decoder'):
        raw_mapping = super().variable_mapping(prefix=prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        
        if self.final_layernorm:
            mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight',
                            'LayerNormFinal.bias': f'{prefix}.LayerNormFinal.bias'})
        if self.with_lm and (not self.tie_word_embeddings):  # 当且仅当未绑定权重的时候
            mapping.update({'lm_head.weight': f'{prefix}.lm_head.weight'})
        return mapping


class Transformer(PreTrainedModelForDecoder):
    '''encoder-decoder结构
    :param tie_word_embeddings: bool, decoder的word_embeddings和lm_head的权重共享
    :param tie_word_embeddings_encoder_decoder: bool, encoder和decoder之间的word_embedding权重共享
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_word_embeddings:bool=False, tie_word_embeddings_encoder_decoder:bool=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self.max_position = kwargs['max_position']
        # decoder的word_embeddings和lm_head的权重共享
        self.tie_word_embeddings = kwargs['tie_word_embeddings'] = tie_word_embeddings
        # encoder和decoder之间的word_embedding权重共享
        self.tie_word_embeddings_encoder_decoder = tie_word_embeddings_encoder_decoder

        self.is_encoder_decoder = True
        self.model_type = 'transformer'

        # encoder
        self.encoder = Encoder(*args, **kwargs)

        # decoder
        self.decoder = Decoder(*args, add_cross_attention=True, **kwargs)

    def tie_weights(self):
        self.decoder.tie_weights()
        # encoder和decoder之间的word_embedding权重共享
        if self.tie_word_embeddings_encoder_decoder:
            assert self.encoder.vocab_size == self.decoder.vocab_size, "To share word embedding, the vocab size of src/tgt shall be the same."
            self.decoder.embeddings.word_embeddings.weight = self.encoder.embeddings.word_embeddings.weight

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

    def _prepare_generation(self, **kwargs):
        if not hasattr(self, 'generation'):
            self.generation = Seq2SeqGeneration(self, **kwargs)
