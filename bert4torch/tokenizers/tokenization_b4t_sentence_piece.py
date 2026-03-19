from .tokenization_b4t_base import TokenizerBase
import unicodedata


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


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))