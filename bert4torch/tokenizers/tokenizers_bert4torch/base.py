# coding=utf-8
'''Tokenization classes
'''

from typing import Any, Literal, List, Union, Dict, Callable
from bert4torch.snippets import truncate_sequences, is_string, lowercase_and_normalize, sequence_padding
from collections import OrderedDict
import torch
import numpy as np
from ..tokenization_utils import Trie


class TokenizerDictOutput(dict):
    '''tokenizer以字典方式输出'''
    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


class TokenizerListOutput(list):
    '''tokenizer以列表方式输出'''
    def to(self, device):
        for i, v in enumerate(self):
            if isinstance(v, torch.Tensor):
                self[i] = v.to(device)
        return self


class TokenizerBase(object):
    """分词器基类
    """
    def __init__(self, token_start:str='[CLS]', token_end:str='[SEP]', token_unk:str='[UNK]', token_pad:str='[PAD]', token_mask:str='[MASK]', 
                 add_special_tokens:Union[str, tuple, list]=None, pre_tokenize:Callable=None, token_translate:Dict=None):
        """参数说明：
        token_unk: 未知词标记
        token_end: 句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
        pad_token: padding填充标记
        token_start: 分类标记，位于整个序列的第一个
        mask_token: mask标记
        pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入pre_tokenize，则先执行pre_tokenize(text)，然后在它的基础上执行原本的tokenize函数；
        token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token。
        """
        self._token_pad = self.pad_token = token_pad
        self._token_unk = self.unk_token = token_unk
        self._token_mask = self.mask_token = token_mask
        self._token_start = self.start_token = token_start
        self._token_end = self.end_token = token_end
        self.never_split = [i for i in [self._token_unk, self._token_end, self._token_pad, self._token_start, self._token_mask] if isinstance(i, str)]
        if add_special_tokens is not None:
            if isinstance(add_special_tokens, (tuple, list)):
                self.never_split.extend(add_special_tokens)
            elif isinstance(add_special_tokens, str):
                self.never_split.append(add_special_tokens)
        self.tokens_trie = self._create_trie(self.never_split)  # trie树主要是为了special_tokens的分词
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            trie.add(token)
        return trie

    def tokenize(self, text:str, maxlen:int=None) -> List[str]:
        """分词函数
        """
        tokens = [self._token_translate.get(token) or token for token in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences([tokens], maxlen, -index)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens:List[str]) -> List[int]:
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def _encode(self, 
                first_text:str, 
                second_text:str=None, 
                maxlen:int=None, 
                pattern:str='S*E*E', 
                truncate_from:Literal['left', 'right']='right', 
                return_offsets:Literal['transformers', True, False]=False):
        """输出文本对应token id和segment id
        """
        first_tokens = self.tokenize(first_text) if is_string(first_text) else first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen is not None:
            # 这里截断思路是优先截断最长的子句
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences([first_tokens, second_tokens], maxlen, index)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        
        encode_output = [first_token_ids, first_segment_ids]
        if return_offsets != False:
            offset = self.rematch(first_text, first_tokens)
            if second_text is not None:
                offset += self.rematch(second_text, second_tokens)
            
            if return_offsets == 'transformers':  # transformers包中tokenizer的形式
                encode_output.append([[0, 0] if not k else [k[0], k[-1]+1] for k in offset])
            else:
                encode_output.append(offset)
        return encode_output

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)
        
    def encode(self, 
               first_texts: Union[str, List[str]], 
               second_texts: Union[str, List[str]] = None, 
               maxlen: int = None, 
               pattern: str = 'S*E*E', 
               truncate_from: Literal['left', 'right'] = 'right', 
               return_offsets: Literal['transformers', True, False] = False, 
               return_tensors: Literal[True, 'pt', 'np'] = None, 
               return_dict: bool = False, 
               **kwargs) -> Union[List[Union[List, np.ndarray, torch.Tensor]], Dict[str, Union[List, np.ndarray, torch.Tensor]]]:
        '''可以处理多条或者单条
        :param first_texts: 需要encode的文本/文本列表
        :param second_texts: 需要encode的文本/文本列表二, 一般需要切分segment_ids使用, 默认未None
        :param maxlen: 允许的最大长度，
        :param pattern: 
        :param truncate_from: 超长时候是截断左侧还是截断右侧, 默认截断右侧right
        :param return_offsets: 是否返回char和token之间的位置映射关系
        :param return_tensors: 是否以tensor的形式返回
        :param return_dict: 是否以dict形式返回

        Returns
        :param input_ids: [CLS] + first_text + [SEP] + second_text
        :param attantion_mask: [1, 1, 1,..., 0, 0, 0]
        '''
        maxlen = maxlen or kwargs.get('max_length')  # 兼容transformers的参数
        return_list = False if isinstance(first_texts, str) else True  # 输入为str时候默认不加btz维度
        first_texts = [first_texts] if isinstance(first_texts, str) else first_texts
        second_texts = [second_texts] if isinstance(second_texts, str) else second_texts

        first_token_ids, first_segment_ids, offsets = [], [], []
        if second_texts is None:
            second_texts = [None] * len(first_texts)
        assert len(first_texts) == len(second_texts), 'first_texts and second_texts should be same length'
        
        # 循环处理每条样本
        for first_text, second_text in zip(first_texts, second_texts):
            outputs = self._encode(first_text, second_text, maxlen, pattern, truncate_from, return_offsets)
            first_token_ids.append(outputs[0])
            first_segment_ids.append(outputs[1])
            if len(outputs) >= 3:
                offsets.append(outputs[2])

        encode_outputs = OrderedDict()
        encode_outputs['input_ids'] = first_token_ids
        encode_outputs['token_type_ids'] = first_segment_ids
        if return_offsets:
            encode_outputs['offset'] = offsets

        if return_tensors in {True, 'pt', 'np'}:
            # 转为tensor
            for key, value in encode_outputs.items():
                if key in {'input_ids', 'token_type_ids'}:
                    encode_outputs[key] = sequence_padding(value, value=self.pad_token_id)
                    if return_tensors == 'pt':
                        encode_outputs[key] = torch.tensor(encode_outputs[key], dtype=torch.long)
        elif not return_list:  # 如果输入是string, 则解胞
            encode_outputs = OrderedDict({key:item[0] for key, item in encode_outputs.items()})
        
        # 是否以字典形式输出
        if return_dict:
            return TokenizerDictOutput(encode_outputs)
        else:
            return TokenizerListOutput([value for value in encode_outputs.values()])

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(int(i)) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError
    
    def rematch(self, text:str, tokens:List[str]) -> List[List]:
        return []
    

