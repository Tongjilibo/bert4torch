'''mlm预测的pipeline
'''
from typing import List, Union, Dict
import numpy as np
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_config_path, sequence_padding
from bert4torch.tokenizers import Tokenizer
from tqdm.autonotebook import trange


class FillMask:
    '''mlm预测
    :param model_path: str, 模型所在文件夹地址
    :param device: str, cpu/cuda
    :param model_config: dict, build_transformer_model时候用到的一些参数
    '''
    def __init__(self, model_path, device=None, **kwargs) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model_path: {model_path} does not exists')
        
        self.model_path = model_path
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model(kwargs)
        self.config = self.model.config
    
    def build_tokenizer(self):
        vocab_path = os.path.join(self.model_path, 'vocab.txt')
        if os.path.exists(vocab_path):
            tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer(self.model_path)
        return tokenizer

    def build_model(self, model_config):
        config_path = get_config_path(self.model_path)  # 获取config文件路径
        checkpoint_path = [os.path.join(self.model_path, i) for i in os.listdir(self.model_path) if i.endswith('.bin')]
        checkpoint_path = checkpoint_path[0] if len(checkpoint_path) == 1 else checkpoint_path
        model_config['with_mlm'] = 'softmax'
        model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **model_config).to(self.device)
        model.eval()
        return model
        
    @torch.inference_mode()
    def predict(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            max_seq_length: int = None
            ):
            
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True
        
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        results = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            batch = self.tokenizer(sentences_batch, maxlen=max_seq_length, return_dict=True)
            results_batch = [{'masked_text': text}  for text in sentences_batch]
            # maskposs
            for text, input_ids in zip(results_batch, batch['input_ids']):
                text['mask_pos'] = [j_i for j_i, j in enumerate(input_ids) if j == self.tokenizer.mask_token_id]

            batch_input = {key: torch.tensor(sequence_padding(value), dtype=torch.long, device=self.device) for key, value in batch.items()}
            mlm_scores = self.model(**batch_input)['mlm_scores']

            for id, (scores, smp) in enumerate(zip(mlm_scores, results_batch)):
                mask_pos = smp['mask_pos']
                pred_token_ids = torch.argmax(scores[mask_pos], dim=-1).cpu().numpy()
                pred_token = [self.tokenizer.decode([j]) for j in pred_token_ids]
                results_batch[id]['pred_token'] = pred_token
                split_texts = results_batch[id]['masked_text'].split(self.tokenizer.mask_token)
                # 替换[MASK]的结果
                filled_text = ''
                count = 0
                for seg in split_texts:
                    if count >= len(pred_token):
                        filled_text += seg
                        continue
                    elif seg == '':
                        filled_text += pred_token[count]
                    else:
                        filled_text += seg + pred_token[count]
                    count += 1
                results_batch[id]['filled_text'] = filled_text
            results.extend(results_batch)
        results = [results[idx] for idx in np.argsort(length_sorted_idx)]

        if input_is_string:
            results = results[0]

        return results