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
    if re.search("transformer\.layers\.[0-9]+\.attention\.query_key_value\.weight", key):
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