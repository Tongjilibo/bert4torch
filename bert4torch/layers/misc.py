import torch
import torch.nn as nn
import numpy as np
from bert4torch.activations import get_activation
from bert4torch.layers.core import LayerNorm
import random
import warnings
import math
from typing import Union


class AdaptiveEmbedding(nn.Module):
    '''Transformer_XL的自适应embedding, 实现不同区间使用不同的维度
    可以实现如高频词用比如1024或512维，低频词用256或64维, 再用Linear层project到相同的维数
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, cutoffs, div_val=1, sample_softmax=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, embedding_size, sparse=sample_softmax > 0))
            if hidden_size != embedding_size:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, embedding_size)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = embedding_size // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, d_emb_i)))

    def forward(self, token_ids):
        if self.div_val == 1:  # 仅有一个embedding
            embed = self.emb_layers[0](token_ids)  # [btz, seq_len, embedding_size]
            if self.hidden_size != self.embedding_size:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = token_ids.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.hidden_size], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed_shape = token_ids.size() + (self.hidden_size,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)

        return embed


class BlockIdentity(nn.Module):
    ''' A placeholder identity operator that is argument-insensitive. '''
    def __init__(self, *args, return_args_index:Union[int, list, set, tuple]=None, return_kwargs_key:Union[str, list, set, tuple]=None, **kwargs):
        '''
        return_args_index: 指定返回的args的index
        return_kwargs_key: 指定返回的kwargs的keys
        '''
        super(BlockIdentity, self).__init__()
        if return_args_index is not None:
            self.return_args_index = return_args_index if isinstance(return_args_index, (tuple,list,set)) else [return_args_index]
        else:
            self.return_args_index = return_args_index
        if return_kwargs_key is not None:
            self.return_kwargs_key = return_kwargs_key if isinstance(return_kwargs_key, (tuple,list,set)) else [return_kwargs_key]
        else:
            self.return_kwargs_key = return_kwargs_key

    def get_args(self, *args):
        new_args = [arg for i, arg in enumerate(args) if i in self.return_args_index]
        if len(new_args) > 1:
            return new_args
        else:
            return new_args[0]
    
    def get_kwargs(self, **kwargs):
        return {k:v for k, v in kwargs.items() if k in self.return_kwargs_key}
    
    def forward(self, *args, **kwargs):
        if self.return_args_index and self.return_kwargs_key:
            return self.get_args(*args), self.get_kwargs(kwargs)
        elif self.return_args_index:
            return self.get_args(*args)
        elif self.return_kwargs_key:
            return self.get_kwargs(kwargs)
        elif (len(args) > 0) and (len(kwargs) > 0):
            return args, kwargs
        elif len(args) > 0:
            return args[0] if len(args) == 1 else args
        elif len(kwargs) > 0:
            return kwargs
        else:
            return None


class BERT_WHITENING():
    ''' 论文：https://arxiv.org/abs/2103.15316
        Github: https://github.com/bojone/BERT-whitening 
    '''
    def __init__(self):
        self.kernel = None
        self.bias = None

    def compute_kernel_bias(self, sentence_vec):
        '''bert-whitening的torch实现
        '''
        vecs = torch.cat(sentence_vec, dim=0)
        self.bias = -vecs.mean(dim=0, keepdims=True)

        cov = torch.cov(vecs.T)  # 协方差
        u, s, vh = torch.linalg.svd(cov)
        W = torch.matmul(u, torch.diag(s**0.5))
        self.kernel = torch.linalg.inv(W.T)
    
    def save_whiten(self, path):
        whiten = {'kernel': self.kernel, 'bias': self.bias}
        torch.save(path, whiten)
        
    def load_whiten(self, path):
        whiten = torch.load(path)
        self.kernel = whiten['kernel']
        self.bias = whiten['bias']

    def transform_and_normalize(self, vecs):
        """应用变换，然后标准化
        """
        if not (self.kernel is None or self.bias is None):
            vecs = (vecs + self.bias).mm(self.kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


class TplinkerHandshakingKernel(nn.Module):
    '''Tplinker的HandshakingKernel实现'''
    def __init__(self, hidden_size, shaking_type, inner_enc_type=''):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        
        # 自行实现的用torch.gather方式来做，避免循环，目前只实现了cat方式
        # tag_ids = [(i, j) for i in range(maxlen) for j in range(maxlen) if j >= i]
        # gather_idx = torch.tensor(tag_ids, dtype=torch.long).flatten()[None, :, None]
        # self.register_buffer('gather_idx', gather_idx)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens

        # def handshaking_kernel(self, last_hidden_state):
        #     '''获取(0,0),(0,1),...,(99,99))对应的序列id
        #     '''
        #     btz, _, hdsz = last_hidden_state.shape
        #     gather_idx = self.gather_idx.repeat(btz, 1, hdsz)
        #     concat_hidden_states = torch.gather(last_hidden_state, dim=1, index=gather_idx)  # [btz, pair_len*2, hdsz]
        #     concat_hidden_states = concat_hidden_states.reshape(btz, -1, 2, hdsz)  # concat方式 [btz, pair_len, 2, hdsz]
        #     shaking_hiddens = torch.cat(torch.chunk(concat_hidden_states, chunks=2, dim=-2), dim=-1).squeeze(-2)  # [btz, pair_len, hdsz*2]
        #     return shaking_hiddens


class MixUp(nn.Module):
    '''mixup方法实现
    
    :param method: str, 可选'embed', ’encoder‘分别表示在embedding和encoder层面做mixup, None表示mix后续处理, ’hidden‘表示对隐含层做mixup
    :param alpha: float
    :param layer_mix: None or int, 需要mix的隐含层index
    '''
    def __init__(self, method='encoder', alpha=1.0, layer_mix=None):
        super().__init__()
        assert method in {'embed', 'encoder', 'hidden', None}
        self.method = method
        self.alpha = alpha
        self.perm_index = None
        self.lam = 0
        self.layer_mix = layer_mix  # 需要mix的隐含层index
    
    def get_perm(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs[self.perm_index]
        elif isinstance(inputs, (list, tuple)):
            return [inp[self.perm_index] if isinstance(inp, torch.Tensor) else inp for inp in inputs]
    
    def mix_up(self, output, output1):
        if isinstance(output, torch.Tensor):
            return self.lam * output + (1.0-self.lam) * output1
        elif isinstance(output, (list, tuple)):
            output_final = []
            for i in range(len(output)):
                if output[i] is None: # conditional_emb=None
                    output_final.append(output[i])
                elif (not output[i].requires_grad) and (output[i].dtype in {torch.long, torch.int}):
                    # 不是embedding形式的
                    output_final.append(torch.max(output[i], output1[i]))
                else:
                    output_final.append(self.lam * output[i] + (1.0-self.lam) * output1[i])
            return output_final
        else:
            raise ValueError('Illegal model output')

    def encode(self, model, inputs):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        self.lam = np.random.beta(self.alpha, self.alpha)
        self.perm_index = torch.randperm(batch_size).to(device)

        if self.method is None:
            output = model(inputs)
            output1 = self.get_perm(output)
            return [output, output1]

        elif self.method == 'encoder':
            output = model(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)

        elif self.method == 'embed':
            output = model.apply_embeddings(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
            # Main
            output_final = model.apply_main_layers(output_final)
            # Final
            output_final = model.apply_final_layers(output_final)
        
        elif self.method == 'hidden':
            if self.layer_mix is None:
                # 这里暂时只考虑encoderLayer, 不考虑decoderLayer和seq2seq模型结构
                try:
                    layer_mix = random.randint(0, len(model.encoderLayer))
                except:
                    warnings.warn('LayerMix random failded')
                    layer_mix = 0
            else:
                layer_mix = self.layer_mix
            
            def apply_on_layer_end(l_i, output):
                if l_i == layer_mix:
                    output1 = self.get_perm(output)
                    return self.mix_up(output, output1)
                else:
                    return output
            model.apply_on_layer_end = apply_on_layer_end
            output_final = model(inputs)
        return output_final
    
    def forward(self, criterion, y_pred, y_true):
        '''计算loss
        '''
        y_true1 = y_true[self.perm_index]
        return self.lam * criterion(y_pred, y_true) + (1 - self.lam) * criterion(y_pred, y_true1)


class ConvLayer(nn.Module):
    '''deberta_v2中使用'''
    def __init__(self, hidden_size, dropout_rate=0.1, layer_norm_eps=1e-12, conv_kernel_size=3, conv_groups=1, conv_act="tanh", **kwargs):
        super().__init__()
        kernel_size = conv_kernel_size
        groups = conv_groups
        self.conv_act = conv_act
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)  # 这里使用bert4torch的LayerNorm会有问题
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = get_activation(self.conv_act)(self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        return output_states


class MultiSampleDropout(nn.Module):
    """multisample dropout (wut): https://arxiv.org/abs/1905.09788"""
    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


class BottleneckAdapterLayer(nn.Module):
    '''BottleneckAdapterLayer'''
    def __init__(self, adapter_input_size, bottleneck_size, adapter_non_linearity='gelu'):
        super().__init__()
        self.adapter_input_size = adapter_input_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = get_activation(adapter_non_linearity)

        # down proj
        self.down_proj = nn.Linear(self.adapter_input_size, self.bottleneck_size)
        # up proj
        self.up_proj = nn.Linear(self.bottleneck_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self, init_mean=0.0, init_std=0.01):
        self.down_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output
    
    
def add_adapter(model, adapter_method='bottleneck', bottlenect_size=64):
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    if adapter_method == 'bottleneck':
        # https://paperswithcode.com/paper/parameter-efficient-transfer-learning-for-nlp
        # https://arxiv.org/pdf/1902.00751v2.pdf
        # 顺序为: Attention --> Adapter --> Add --> LN --> FeedForward --> Adapter --> Add --> LayerNorm
        
        try:
            layers = model.encoderLayer
        except:
            layers = model.decoderLayer
        
        # TODO: 这里需要测试
        for layer_id in range(model.num_hidden_layers):
            transformer_layer = layers[layer_id].multiHeadAttention.o
            out_featuers = transformer_layer.out_features
            adapter1 = BottleneckAdapterLayer(out_featuers, bottleneck_size=bottlenect_size)
            layers[layer_id].multiHeadAttention.o = nn.Sequential(transformer_layer, adapter1)

            transformer_layer = layers[layer_id].feedForward
            out_featuers = transformer_layer.outputDense.out_features
            adapter2 = BottleneckAdapterLayer(out_featuers, bottleneck_size=bottlenect_size)
            layers[layer_id].feedForward = nn.Sequential(transformer_layer, adapter2)
    # 待新增其余类型adapter
    else:
        pass
    return model


class NormHead(nn.Module):
    '''normalized lm_head，目前是Baichuan2使用'''
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x