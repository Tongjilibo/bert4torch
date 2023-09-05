from bert4torch.models.internlm import InternLM


class Qwen(InternLM):
    '''通义千问: https://github.com/QwenLM/Qwen-7B
    1）FeedForward和Llama一致，三个dense层
    2）除了qkv有bias，其余均没有bias
    3) 和InternLM基本一致，唯一的差别是InternLM的multiHeadAttention.o有bias
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.decoderLayer:
            layer.multiHeadAttention.o.register_parameter('bias', None)

    def load_variable(self, state_dict, name, prefix='qwen'):
        return super().load_variable(state_dict, name, prefix=prefix)

    def variable_mapping(self, prefix='qwen'):
        # 映射到权重格式
        return super().variable_mapping(prefix=prefix)
