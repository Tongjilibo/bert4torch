from bert4torch.models.internlm import InternLM


class Qwen(InternLM):
    '''通义千问: https://github.com/QwenLM/Qwen-7B
    1）FeedForward和Llama一致，三个dense层
    2）除了qkv有bias，其余均没有bias
    3) 和InternLM基本一致，唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = 'qwen'
        for layer in self.decoderLayer:
            layer.multiHeadAttention.o.register_parameter('bias', None)
