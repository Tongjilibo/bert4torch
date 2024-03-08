'''句向量pipeline
   类似sentence-transformers的调用方式`model.encode(sentences)`
'''
from typing import List, Union, Dict
import numpy as np
import torch
from bert4torch.snippets import get_pool_emb, sequence_padding
from tqdm.autonotebook import trange
from .base import PipeLineBase


class Text2Vec(PipeLineBase):
    '''句向量, 目前支持m3e, bge, simbert, text2vec-base-chinese
    :param checkpoint_path: str, 模型所在文件夹地址
    :param device: str, cpu/cuda
    :param model_config: dict, build_transformer_model时候用到的一些参数
    '''
    def __init__(self, checkpoint_path, device=None, **kwargs) -> None:
        super().__init__(checkpoint_path, device, **kwargs)
        self.pool_strategy = self.config.get('pool_strategy', 'cls')
        
    @torch.inference_mode()
    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            pool_strategy=None,
            custom_layer=None,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            max_seq_length: int = None
            ):
        
        if convert_to_tensor:
            convert_to_numpy = False
    
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True
        
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            batch = self.tokenizer(sentences_batch, maxlen=max_seq_length)
            batch_input = [torch.tensor(sequence_padding(item), dtype=torch.long, device=self.device) for item in batch]
            output = self.model(batch_input)

            last_hidden_state = output.get('last_hidden_state')
            pooler = output.get('pooled_output')
            if isinstance(batch_input, list):
                attention_mask = (batch_input[0] != self.tokenizer._token_pad_id).long()
            elif isinstance(batch_input, torch.Tensor):
                attention_mask = (batch_input != self.tokenizer._token_pad_id).long()
            else:
                raise TypeError('Args `batch_input` only support list(tensor)/tensor format')
            pool_strategy = pool_strategy or self.pool_strategy
            embs = get_pool_emb(last_hidden_state, pooler, attention_mask, pool_strategy, custom_layer)
            if normalize_embeddings:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            if convert_to_numpy:
                embs = embs.cpu()
            all_embeddings.extend(embs)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings    