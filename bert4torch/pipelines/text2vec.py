from typing import List, Union, Dict
import numpy as np
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_pool_emb, sequence_padding
from bert4torch.tokenizers import Tokenizer
from tqdm.autonotebook import trange


class Text2Vec:
    '''句向量'''
    def __init__(self, model_path=None, vocab_path=None, config_path=None, checkpoint_path=None, device='cpu', model_config=None) -> None:
        if model_path is not None:
            vocab_path = vocab_path or os.path.join(model_path, 'vocab.txt')
            config_path = config_path or os.path.join(model_path, 'config.json')
            checkpoint_path = checkpoint_path or [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith('.bin')]
        self.tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        model_config = model_config or dict()
        self.model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **model_config).to(device)
        self.model.eval()
        self.device = device
    
    @torch.inference_mode()
    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            pool_strategy='cls',
            custom_layer=None,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = True,
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
            attention_mask = (last_hidden_state != self.tokenizer._token_pad_id).long()
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