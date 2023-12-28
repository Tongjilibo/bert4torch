from typing import List, Union, Dict
import numpy as np
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_pool_emb, sequence_padding, JsonConfig
from bert4torch.tokenizers import Tokenizer
from tqdm.autonotebook import trange


class Text2Vec:
    '''句向量, 目前支持m3e, bge, simbert, text2vec-base-chinese
    :param model_path: str, 模型所在文件夹地址
    :param device: str, cpu/cuda
    :param model_config: dict, build_transformer_model时候用到的一些参数
    '''
    def __init__(self, model_path, device='cpu', **kwargs) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model_path: {model_path} does not exists')
        
        self.model_path = model_path
        self.device = device
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model(kwargs)
        self.pool_strategy = self.config.get('pool_strategy', 'cls')
    
    def build_tokenizer(self):
        vocab_path = os.path.join(self.model_path, 'vocab.txt')
        if os.path.exists(vocab_path):
            tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer(self.model_path)
        return tokenizer

    def build_model(self, model_config):
        config_path = None
        for _config in ['bert4torch_config.json', 'config.json']:
            config_path = os.path.join(self.model_path, _config)
            if os.path.exists(config_path):
                break
        if config_path is None:
            raise FileNotFoundError('Config file not found')

        self.config = JsonConfig(config_path)

        checkpoint_path = [os.path.join(self.model_path, i) for i in os.listdir(self.model_path) if i.endswith('.bin')]
        checkpoint_path = checkpoint_path[0] if len(checkpoint_path) == 1 else checkpoint_path
        model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **model_config).to(self.device)
        model.eval()
        return model
        
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