import torch
import re

dir_path = 'F:/Projects/pretrain_ckpt/chatglm/6B/'
state_dict = {}
new_weights = {}

# 把所有权重读入同一个文件
for i in range(1, 9):
    file_path = f"pytorch_model-0000{i}-of-00008.bin"
    state_dict_tmp = torch.load(dir_path+file_path)
    state_dict.update(state_dict_tmp)
    del state_dict_tmp

for key, value in state_dict.items():
    if key in {"lm_head.weight", "transformer.word_embeddings.weight"}:
        new_weights[key] = value[20000:]  # 前2万个token都是图像相关的，因此删减掉
    elif re.search("transformer\.layers\.[0-9]+\.attention\.query_key_value\.weight", key):
        l = re.findall('[0-9]+', key)[0]
        tensor_list = torch.split(value, 128, 0)
        q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
        q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
        new_weights[f'transformer.layers.{l}.attention.self.query.weight'] = q
        new_weights[f'transformer.layers.{l}.attention.self.key.weight'] = k
        new_weights[f'transformer.layers.{l}.attention.self.value.weight'] = v
    elif re.search("transformer\.layers\.[0-9]+\.attention\.query_key_value\.bias", key):
        l = re.findall('[0-9]+', key)[0]
        tensor_list = torch.split(value, 128, 0)
        q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
        q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
        new_weights[f'transformer.layers.{l}.attention.self.query.bias'] = q
        new_weights[f'transformer.layers.{l}.attention.self.key.bias'] = k
        new_weights[f'transformer.layers.{l}.attention.self.value.bias'] = v
    else:
        new_weights[key] = value

torch.save(new_weights, dir_path + 'bert4torch_pytorch_model.bin')

'''
{
  "hidden_act": "gelu", 
  "bos_token_id": 130004,
  "eos_token_id": 130005,
  "hidden_size": 4096,
  "intermediate_size": 16384,
  "layer_norm_eps": 1e-05,
  "max_sequence_length": 2048,
  "num_attention_heads": 32,
  "num_hidden_layers": 28,
  "position_encoding_2d": true,
  "torch_dtype": "float16",
  "vocab_size": 130528,
  "segment_vocab_size": 0
}
'''