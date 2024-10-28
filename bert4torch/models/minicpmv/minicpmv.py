import math
from typing import Union
import torch
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler
from bert4torch.models.qwen import Qwen2
from bert4torch.models.llama import LLaMA
from bert4torch.snippets import DottableDict
import inspect


class MiniCPMV(Qwen2):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        self.vlm_tgt_sizes = True if 'tgt_sizes' in inspect.signature(model).parameters else False
        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def get_vllm_embedding(self, data):
        if data.get('vision_hidden_states') is None:
            dtype = self.embeddings.word_embeddings.weight.dtype
            device = self.embeddings.word_embeddings.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        inputs_ = {'patch_attention_mask': patch_attn_mask[start_idx:end_idx]}
                        if self.vlm_tgt_sizes:
                            inputs_['tgt_sizes'] = tgt_sizes[start_idx:end_idx]
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], **inputs_).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    inputs_ = {'patch_attention_mask': patch_attn_mask}
                    if self.vlm_tgt_sizes:
                        inputs_['tgt_sizes'] = tgt_sizes
                    vision_embedding = self.vpm(all_pixel_values, **inputs_).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if self.embeddings.emb_scale != 1:
            vllm_embedding = self.embeddings.word_embeddings(data['input_ids']) * self.embeddings.emb_scale
        else:
            vllm_embedding = self.embeddings.word_embeddings(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states
    
    def load_variable(self, variable, old_key, new_key):
        if old_key in {'llm.embeddings.word_embeddings.weight', 'llm.lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'embeddings.word_embeddings.weight': 'llm.model.embed_tokens.weight',
            'lm_head.weight': 'llm.lm_head.weight',
            'LayerNormFinal.weight': 'llm.model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.q.weight': f'llm.model.layers.{i}.self_attn.q_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.q.bias': f'llm.model.layers.{i}.self_attn.q_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'llm.model.layers.{i}.self_attn.k_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.k.bias': f'llm.model.layers.{i}.self_attn.k_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'llm.model.layers.{i}.self_attn.v_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.v.bias': f'llm.model.layers.{i}.self_attn.v_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'llm.model.layers.{i}.self_attn.o_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'llm.model.layers.{i}.self_attn.o_proj.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'llm.model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'llm.model.layers.{i}.mlp.gate_proj.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'llm.model.layers.{i}.mlp.up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'llm.model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'llm.model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping

    def apply_embeddings(self, *inputs:Union[tuple, list], **model_kwargs):
        ''' 准备进embedding层的一些输入'''
        input_ids, _, _, _, _, attention_mask, model_kwargs = self.preprare_embeddings_inputs(*inputs, **model_kwargs)
        assert input_ids is not None

        model_kwargs.update({"input_ids": input_ids})

        position_ids = model_kwargs.get("position_ids")
        if position_ids.dtype != torch.int64:
            model_kwargs["position_ids"] = position_ids.long()

        vllm_embedding, _ = self.get_vllm_embedding(model_kwargs)

        # 进入embedding层
        model_kwargs.update({'hidden_states': vllm_embedding, 'attention_mask':attention_mask})
        
        return model_kwargs

    def prepare_inputs_for_generation(self, *inputs, **states):
        return super().prepare_inputs_for_generation(*inputs, **states)

    def get_states(self, kwargs):
        new_kwargs = super().get_states(kwargs)
        for key in ['attention_mask', 'pixel_values', 'image_bound', 'tgt_sizes', 'vision_hidden_states']:
            if key in kwargs:
                new_kwargs[key] = kwargs[key]
        return new_kwargs
    

class MiniCPMLlama3V(LLaMA):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )
    
    def init_vision_module(self):
        from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
        model = Idefics2VisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)
        self.vlm_tgt_sizes = True if 'tgt_sizes' in inspect.signature(model).parameters else False

        return model
    
    def variable_mapping(self):
        mapping = super().variable_mapping()
        mapping = {k:v for k,v in mapping.items() if '.bias' not in k}
        return mapping