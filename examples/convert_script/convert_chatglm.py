'''chatglm-6b的转换脚本
default:  https://huggingface.co/THUDM/chatglm-6b
int4:     https://huggingface.co/THUDM/chatglm-6b-int4
int8:     https://huggingface.co/THUDM/chatglm-6b-int8
chatglm2: https://huggingface.co/THUDM/chatglm2-6b
chatglm2-int4: https://huggingface.co/THUDM/chatglm2-6b-int4
'''
import torch
import re
import json


choice = 'default'  # default, int4, int8, v1.1.0, chatglm2, chatglm2-int4
layer_prefix = 'decoderLayer'  # v0.3.0(含)之前使用 encoderLayer ，之后使用 decoderLayer

if choice == 'default':
    dir_path = 'E:/pretrain_ckpt/chatglm/6B/'
elif choice == 'v1.1.0':
    dir_path = 'E:/pretrain_ckpt/chatglm/6B-v1_1_0/'
elif choice == 'int4':
    dir_path = 'E:/pretrain_ckpt/chatglm/6B-int4/'
elif choice == 'int8':
    dir_path = 'E:/pretrain_ckpt/chatglm/6B-int8/'
elif choice == 'chatglm2':
    dir_path = 'E:/pretrain_ckpt/chatglm2/6B/'
elif choice == 'chatglm2-int4':
    dir_path = 'E:/pretrain_ckpt/chatglm2/6B-int4/'
else:
    raise ValueError(f'{choice} not in pre maintained choices')
    

def trans_chatglm(state_dict_tmp):
    '''chatglm权重转换'''
    new_weights = {}
    for key, value in state_dict_tmp.items():
        # 旧逻辑是删除前20000个token，但是清华官方repo在20230406时候清理了，这里就改为不删减
        # if key in {"lm_head.weight", "transformer.word_embeddings.weight"}:
        #     new_weights[key] = value[20000:]  # 前2万个token都是图像相关的，因此删减掉
        #     continue
        
        # int4和int8专属
        if re.search("transformer\.layers\.[0-9]+\.attention\.query_key_value\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            tensor_list = torch.split(value, 128, 0)
            q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
            q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.q.weight_scale'] = q
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.k.weight_scale'] = k
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.v.weight_scale'] = v
        elif re.search("transformer\.layers\.[0-9]+\.attention\.dense\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.o.weight_scale'] = value
        elif re.search("transformer\.layers\.[0-9]+\.mlp\.dense_h_to_4h\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.feedForward.intermediateDense.weight_scale'] = value
        elif re.search("transformer\.layers\.[0-9]+\.mlp\.dense_4h_to_h\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.feedForward.outputDense.weight_scale'] = value

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
    return new_weights

def trans_chatglm2(state_dict_tmp):
    '''chatglm2权重转换'''
    new_weights = {}
    for key, value in state_dict_tmp.items():
        # int4和int8专属
        if re.search("transformer\.encoder\.layers\.[0-9]+\.self_attention\.query_key_value\.weight_scale", key):
            q, k, v = torch.split(value, [4096, 256, 256], 0)
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.q.weight_scale'] = q
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.k.weight_scale'] = k
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.v.weight_scale'] = v
        elif re.search("transformer\.encoder\.layers\.[0-9]+\.self_attention\.dense\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.multiHeadAttention.o.weight_scale'] = value
        elif re.search("transformer\.encoder\.layers\.[0-9]+\.mlp\.dense_h_to_4h\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.feedForward.intermediateDense.weight_scale'] = value
        elif re.search("transformer\.encoder\.layers\.[0-9]+\.mlp\.dense_4h_to_h\.weight_scale", key):
            l = re.findall('[0-9]+', key)[0]
            new_weights[f'{layer_prefix}.{l}.feedForward.outputDense.weight_scale'] = value

        elif re.search("transformer\.encoder\.layers\.[0-9]+\.self_attention\.query_key_value\.weight", key):
            l = re.findall('[0-9]+', key)[0]
            q, k, v = torch.split(value, [4096, 256, 256], 0)
            new_weights[f'transformer.layers.{l}.attention.self.query.weight'] = q
            new_weights[f'transformer.layers.{l}.attention.self.key.weight'] = k
            new_weights[f'transformer.layers.{l}.attention.self.value.weight'] = v
        elif re.search("transformer\.encoder\.layers\.[0-9]+\.self_attention\.query_key_value\.bias", key):
            l = re.findall('[0-9]+', key)[0]
            q, k, v = torch.split(value, [4096, 256, 256], 0)
            new_weights[f'transformer.layers.{l}.attention.self.query.bias'] = q
            new_weights[f'transformer.layers.{l}.attention.self.key.bias'] = k
            new_weights[f'transformer.layers.{l}.attention.self.value.bias'] = v
        elif re.search("transformer.embedding.word_embeddings.weight", key):
            new_weights['transformer.word_embeddings.weight'] = value
        elif re.search("transformer\.encoder\.layers\.[0-9]+\.self_attention\.dense\.weight", key):
            new_weights[f'transformer.layers.{l}.attention.dense.weight'] = value
        elif re.search("transformer\.output_layer\.weight", key):
            new_weights[f'lm_head.weight'] = value
        elif re.search('rotary_pos_emb', key):
            pass
        else:
            key = key.replace('.encoder.', '.')
            new_weights[key] = value
    return new_weights

if choice in {'default', 'v1.1.0'}:
    # 依次读入权重
    for i in range(1, 9):
        file_path = f"pytorch_model-0000{i}-of-00008.bin"
        state_dict_tmp = torch.load(dir_path+file_path)

        # 保存成多个文件
        new_weights = trans_chatglm(state_dict_tmp)
        torch.save(new_weights, dir_path + f'bert4torch_pytorch_model_{i}.bin')

elif choice in {'int8', 'int4'}:
    state_dict_tmp = torch.load(dir_path+'pytorch_model.bin')
    new_weights = trans_chatglm(state_dict_tmp)
    torch.save(new_weights, dir_path + f'bert4torch_pytorch_model.bin')

elif choice == 'chatglm2':
    for i in range(1, 8):
        file_path = f"pytorch_model-0000{i}-of-00007.bin"
        state_dict_tmp = torch.load(dir_path+file_path)

        # 保存成多个文件
        new_weights = trans_chatglm2(state_dict_tmp)
        torch.save(new_weights, dir_path + f'bert4torch_pytorch_model_{i}.bin')

elif choice == 'chatglm2-int4':
    file_path = f"pytorch_model.bin"
    state_dict_tmp = torch.load(dir_path+file_path)

    # 保存成多个文件
    new_weights = trans_chatglm2(state_dict_tmp)
    torch.save(new_weights, dir_path + f'bert4torch_pytorch_model.bin')


# config配置
if choice in {'default', 'v1.1.0'}:
    config = \
    {
    "hidden_act": "gelu_new", 
    "bos_token_id": 130004,
    "eos_token_id": 130005,
    "mask_token_id": 130000,
    "gmask_token_id": 130001,
    "pad_token_id": 3,
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "layer_norm_eps": 1e-05,
    "max_sequence_length": 2048,
    "num_attention_heads": 32,
    "num_hidden_layers": 28,
    "position_encoding_2d": True,
    "torch_dtype": "float16",
    "vocab_size": 130528,
    "segment_vocab_size": 0,
    "skip_init": True,
    "rope_rank": "updown",
    "tie_emb_prj_weight": False
    }

# int4 config
elif choice == 'int4':
    config = \
    {
        "hidden_act": "gelu_new", 
        "bos_token_id": 130004,
        "eos_token_id": 130005,
        "mask_token_id": 130000,
        "gmask_token_id": 130001,
        "pad_token_id": 3,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "layer_norm_eps": 1e-05,
        "max_sequence_length": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 28,
        "position_encoding_2d": True,
        "torch_dtype": "float16",
        "vocab_size": 130528,
        "segment_vocab_size": 0,
        "skip_init": True,
        "rope_rank": "updown",
        "quantization_bit": 4,
        "quantization_method": "cpm_kernels",
        "target_modules": ["q", "k", "v", "o", "intermediateDense", "outputDense"],
        "tie_emb_prj_weight": False
    }

# int8 config
elif choice == 'int8':
    config = \
    {
    "hidden_act": "gelu_new", 
    "bos_token_id": 130004,
    "eos_token_id": 130005,
    "mask_token_id": 130000,
    "gmask_token_id": 130001,
    "pad_token_id": 3,
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "layer_norm_eps": 1e-05,
    "max_sequence_length": 2048,
    "num_attention_heads": 32,
    "num_hidden_layers": 28,
    "position_encoding_2d": True,
    "torch_dtype": "float16",
    "vocab_size": 130528,
    "segment_vocab_size": 0,
    "skip_init": True,
    "rope_rank": "updown",
    "quantization_bit": 8,
    "quantization_method": "cpm_kernels",
    "target_modules": ["q", "k", "v", "o", "intermediateDense", "outputDense"],
    "tie_emb_prj_weight": False
    }

# chatglm2
elif choice == 'chatglm2':
    config = \
    {
    "hidden_act": "swiglu", 
    "hidden_size": 4096,
    "intermediate_size": 13696,
    "layer_norm_eps": 1e-05,
    "max_sequence_length": 32768,
    "num_attention_heads": 32,
    "num_hidden_layers": 28,
    "vocab_size": 65024,
    "segment_vocab_size": 0,
    "multi_query_group_num": 2,
    "skip_init": True,
    "tie_emb_prj_weight": False,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "rmsnorm": True,
    "rope_rank": "updown",
    "position_encoding_2d_v2": True,
    "flash_attention": True
    }

# chatglm2-int4
elif choice == 'chatglm2-int4':
    config = \
    {
    "hidden_act": "swiglu", 
    "hidden_size": 4096,
    "intermediate_size": 13696,
    "layer_norm_eps": 1e-05,
    "max_sequence_length": 32768,
    "num_attention_heads": 32,
    "num_hidden_layers": 28,
    "vocab_size": 65024,
    "segment_vocab_size": 0,
    "multi_query_group_num": 2,
    "skip_init": True,
    "tie_emb_prj_weight": False,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "rmsnorm": True,
    "rope_rank": "updown",
    "position_encoding_2d_v2": True,
    "flash_attention": True,
    "quantization_bit": 4,
    "quantization_method": "cpm_kernels",
    "target_modules": ["q", "k", "v", "o", "intermediateDense", "outputDense"]
    }

with open(dir_path+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))
