'''mlm预测的pipeline
'''
from typing import List, Union, Dict
import numpy as np
import torch
from bert4torch.snippets import sequence_padding
from .base import PipeLineBase
from tqdm.autonotebook import trange


class FillMask(PipeLineBase):
    '''mlm预测
    :param checkpoint_path: str, 模型所在文件夹地址
    :param device: str, cpu/cuda
    :param model_config: dict, build_transformer_model时候用到的一些参数

    Examples:
    ```python
    >>> model = FillMask('/home/pretrain_ckpt/bert/bert-base-chinese')
    >>> res = model.predict(["今天[MASK]气不错，[MASK]情很好", '[MASK]学技术是第一生产力'])
    ```
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, with_mlm='softmax', **kwargs)

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