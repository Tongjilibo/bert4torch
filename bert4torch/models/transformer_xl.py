from bert4torch.models.bert import BERT
from bert4torch.snippets import insert_arguments, delete_arguments
from bert4torch.layers import AdaptiveEmbedding, XlnetPositionsEncoding
from bert4torch.layers import BlockIdentity, XlnetLayer
from torch import nn
import torch
import copy


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
        kwargs.update({'p_bias': 'MultiHeadAttention'})
        self.attn_type = kwargs.pop('attn_type', 0)  # pop出来防止影响内部attn_type
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        super().__init__(*args, **kwargs)

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
        self.encoderLayer = nn.ModuleList([XlnetLayer(layer_idx=layer_idx, r_s_bias=None, **self.get_kw('r_w_bias', 'r_r_bias', *self._layer_args, **kwargs)) 
                                           if layer_idx in self.keep_hidden_layers else BlockIdentity() for layer_idx in range(self.num_hidden_layers)])

        # 映射
        if self.with_lm:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.model_type = 'transformer_xl'

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

        # ========================= token_ids =========================
        index_ = 0
        if model_kwargs.get('input_ids') is not None:
            token_ids = model_kwargs['input_ids']
        elif model_kwargs.get('token_ids') is not None:
            token_ids = model_kwargs['token_ids']
        else:
            token_ids = inputs[0]
            index_ += 1

        self.mems = self.init_mems(token_ids.size(0))  # 生成mems
        # 精简后embeddings中只计算word_emdedding
        word_emb = self.dropout(self.embeddings(token_ids))
        index_ = 1
        btz, qlen = token_ids.shape[:2]  # query长度
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen
        # 相对位置编码
        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)
        
        # ========================= segment_ids =========================
        # segment embedding
        if model_kwargs.get('segment_ids') is not None:
            segment_ids = model_kwargs['segment_ids']
        elif model_kwargs.get('token_type_ids') is not None:
            segment_ids = model_kwargs['token_type_ids']
        elif self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None
        if segment_ids is not None:
            if mlen > 0:
                mem_pad = torch.zeros([btz, mlen], dtype=torch.long, device=word_emb.device)
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            # `1` indicates not in the same segment [qlen x klen x bsz]
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()

        if self.attn_type in {'uni', 0}:  # 兼容transformer_xl的设置: 0
            non_tgt_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == 'bi':
            attention_mask = (token_ids != self.pad_token_id).long().unsqueeze(1).unsqueeze(2)
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
            outputs = self.layer_forward(layer_module, model_kwargs)
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
    
    def load_variable(self, variable, ckpt_key, model_key):
        # 这里由于预训练模型使用了AdapterEmbedding，因此暂不支持
        if (self.keep_tokens is not None) or (self.compound_tokens is not None):
            raise ValueError('Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL')
        return variable

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            qkv_net = state_dict.pop(f'transformer.layers.{i}.dec_attn.qkv_net.weight')
            for k, v in zip(['q', 'k', 'v'], qkv_net.chunk(3, dim=0)):
                state_dict[f'encoderLayer.{i}.multiHeadAttention.{k}.weight'] = v
        return state_dict
    
    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            qkv = []
            model_key = 'encoderLayer.{}.multiHeadAttention.{}.weight'
            for i_k in ['q', 'k', 'v']:
                if model_key.format(i, i_k) in state_dict:
                    qkv.append(state_dict.pop(model_key.format(i, i_k)))
            if qkv:
                state_dict[f'transformer.layers.{i}.dec_attn.qkv_net.weight'] = torch.cat(qkv)

        return state_dict

    def variable_mapping(self):
        mapping = {
            'embeddings.emb_layers.0.weight': 'transformer.word_emb.emb_layers.0.weight',
            'embeddings.emb_layers.1.weight': 'transformer.word_emb.emb_layers.1.weight',
            'embeddings.emb_layers.2.weight': 'transformer.word_emb.emb_layers.2.weight',
            'embeddings.emb_layers.3.weight': 'transformer.word_emb.emb_layers.3.weight',
            'embeddings.emb_projs.0': 'transformer.word_emb.emb_projs.0',
            'embeddings.emb_projs.1': 'transformer.word_emb.emb_projs.1',
            'embeddings.emb_projs.2': 'transformer.word_emb.emb_projs.2',
            'embeddings.emb_projs.3': 'transformer.word_emb.emb_projs.3',
            }

        for i in range(self.num_hidden_layers):
            mapping.update({
                f'encoderLayer.{i}.multiHeadAttention.r_r_bias': f'transformer.layers.{i}.dec_attn.r_r_bias',
                f'encoderLayer.{i}.multiHeadAttention.r_w_bias': f'transformer.layers.{i}.dec_attn.r_w_bias',
                f'encoderLayer.{i}.multiHeadAttention.o.weight': f'transformer.layers.{i}.dec_attn.o_net.weight',
                f'encoderLayer.{i}.attnLayerNorm.weight': f'transformer.layers.{i}.dec_attn.layer_norm.weight',
                f'encoderLayer.{i}.attnLayerNorm.bias': f'transformer.layers.{i}.dec_attn.layer_norm.bias',
                f'encoderLayer.{i}.multiHeadAttention.r.weight': f'transformer.layers.{i}.dec_attn.r_net.weight',
                f'encoderLayer.{i}.feedForward.intermediateDense.weight': f'transformer.layers.{i}.pos_ff.CoreNet.0.weight',
                f'encoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.layers.{i}.pos_ff.CoreNet.0.bias',
                f'encoderLayer.{i}.feedForward.outputDense.weight': f'transformer.layers.{i}.pos_ff.CoreNet.3.weight',
                f'encoderLayer.{i}.feedForward.outputDense.bias': f'transformer.layers.{i}.pos_ff.CoreNet.3.bias',
                f'encoderLayer.{i}.ffnLayerNorm.weight': f'transformer.layers.{i}.pos_ff.layer_norm.weight',
                f'encoderLayer.{i}.ffnLayerNorm.bias': f'transformer.layers.{i}.pos_ff.layer_norm.bias',
            })
        return mapping