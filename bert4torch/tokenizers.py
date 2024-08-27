# coding=utf-8
'''Tokenization classes
'''

import collections
import logging
from typing import Any, Literal, List, Union, Dict, Callable
import unicodedata
from io import open
from bert4torch.snippets import truncate_sequences, is_string, lowercase_and_normalize, sequence_padding
import re
import six
from collections import OrderedDict
import torch
import numpy as np


logger = logging.getLogger(__name__)
is_py2 = six.PY2

def load_vocab(dict_path:str, encoding:str="utf-8", simplified:bool=False, startswith:List=None):
    """加载词典文件到dict"""
    token_dict = collections.OrderedDict()
    index = 0
    with open(dict_path, "r", encoding=encoding) as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token_dict[token] = index
            index += 1

    if simplified:  # 过滤冗余部分token，如[unused1]
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


def whitespace_tokenize(text):
    """去除文本中的空白符"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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
    

class Tokenizer(TokenizerBase):
    """Bert原生分词器
    """
    def __init__(self, token_dict:Union[str, dict], do_lower_case:bool=True, do_basic_tokenize:bool=True, do_tokenize_unk:bool=False, **kwargs):
        """
        :param token_dict: 词典文件
        :param do_lower_case: 是否转换成小写
        :param do_basic_tokenize: 分词前，是否进行基础的分词
        :param do_tokenize_unk: 分词后，是否生成[UNK]标记，还是在encode阶段生成
        """
        super(Tokenizer, self).__init__(**kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)

        self._do_lower_case = do_lower_case
        self._vocab_size = len(token_dict)
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}

        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=self.never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self._token_dict, unk_token=self._token_unk, do_tokenize_unk=do_tokenize_unk)
        # 以下写在外面是方便有代码提示
        self._token_pad_id = self.pad_token_id = None
        self._token_unk_id = self.unk_token_id = None
        self._token_mask_id = self.mask_token_id = None
        self._token_start_id = self.start_token_id = None
        self._token_end_id = self.end_token_id = None
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
                setattr(self, '%s_token_id' % token, _token_id)
            except:
                delattr(self, '_token_%s_id' % token)
                delattr(self, '%s_token_id' % token)

    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        # 以下pre_tokenizer逻辑参考bert4keras
        if self._do_lower_case:
            text = lowercase_and_normalize(text, never_split=self.never_split)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens

        # 以下逻辑参考pytorch版本bert分词器自己的
        text_pieces = self.tokens_trie.split(text)  # 新增逻辑，主要是special_tokens的分词
        split_tokens = []
        for text_piece in text_pieces:
            if not text_piece:
                continue
            elif text_piece in self._token_dict:
                split_tokens.append(text_piece)
            elif self.do_basic_tokenize:
                for token in self.basic_tokenizer.tokenize(text_piece):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
            else:
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(text_piece))
        return split_tokens

    def token_to_id(self, token):
        """token转为vocab中的id"""
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id):
        """id转为词表中的token"""
        return self._token_dict_inv[id]

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def _is_redundant(token):
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if (
                    Tokenizer._is_cjk_character(ch) or
                    Tokenizer._is_punctuation(ch)
                ):
                    return True

    def rematch(self, text:str, tokens:List[str]) -> List[List]:
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if is_py2:
            text = unicode(text)

        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = lowercase_and_normalize(ch, self.never_split)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text:str):
        """文本切分成token"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100, do_tokenize_unk=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_tokenize_unk = do_tokenize_unk

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token if self.do_tokenize_unk else token)  # 超长
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break

                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if self.do_tokenize_unk and is_bad:  # 是否在tokenize阶段转UNK
                output_tokens.append(self.unk_token)
            elif (not self.do_tokenize_unk) and is_bad:
                output_tokens.append(substr)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class SpTokenizer(TokenizerBase):
    """基于SentencePiece模型的封装，使用上跟Tokenizer基本一致。
    """
    def __init__(self, sp_model_path, remove_space=True, keep_accents=False, do_lower_case=False, **kwargs):
        super(SpTokenizer, self).__init__(**kwargs)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        self._vocab_size = self.sp_model.get_piece_size()
        self._token_pad = self.id_to_token(self.sp_model.pad_id())
        self._token_unk = self.id_to_token(self.sp_model.unk_id())
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.do_lower_case = do_lower_case

        # pad和unk肯定存在，改动是为了处理llama中pad_id是-1的情况
        self._token_pad_id = self.pad_token_id = self.sp_model.pad_id()
        self._token_unk_id = self.unk_token_id = self.sp_model.unk_id()
        self._token_mask_id = self.mask_token_id = None
        self._token_start_id = self.start_token_id = None
        self._token_end_id = self.end_token_id = None
        for token in ['mask', 'start', 'end']:
            try:
                _token = getattr(self, '_token_%s' % token)
                _token_id = self.sp_model.piece_to_id(_token)
                setattr(self, '_token_%s_id' % token, _token_id)
                setattr(self, '%s_token_id' % token, _token_id)
            except:
                delattr(self, '_token_%s_id' % token)
                delattr(self, '%s_token_id' % token)

    def preprocess_text(self, inputs):
        '''从transformers包的tokenization_xlnet移植过来，主要区别是对标点符号的处理
        '''
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs
    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self.sp_model.piece_to_id(token)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        if (0 <= i) and (i < self._vocab_size):
            return self.sp_model.id_to_piece(i)
        else:
            return ''

    def decode(self, ids):
        """转为可读文本
        """
        tokens = [self._token_translate_inv.get(token) or token for token in self.ids_to_tokens(ids)]
        text = self.sp_model.decode_pieces(tokens)
        return convert_to_unicode(text)

    def _tokenize(self, text):
        """基本分词函数
        """
        if self._pre_tokenize is not None:
            text = ' '.join(self._pre_tokenize(text))

        text = self.preprocess_text(text)  # 是否去空格，转符号，转小写
        tokens = self.sp_model.encode_as_pieces(text)
        return tokens

    def _is_special(self, i):
        """判断是不是有特殊含义的符号
        """
        return self.sp_model.is_control(i) or \
            self.sp_model.is_unknown(i) or \
            self.sp_model.is_unused(i)

    def _is_decodable(self, i):
        """判断是否应该被解码输出
        """
        return (i < self._vocab_size) and not self._is_special(i)


class Trie:
    """直接从transformer的tokenization_utils.py中移植, 主要是为了special_tokens分词
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str):
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens
