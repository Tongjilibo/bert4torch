'''输出句向量的pipeline
   类似sentence-transformers的调用方式`model.encode(sentences)`
'''
from typing import List, Union, Dict
import numpy as np
import torch
from bert4torch.snippets import get_pool_emb, PoolStrategy, log_warn
from tqdm.autonotebook import trange
from .base import PipeLineBase


class Text2Vec(PipeLineBase):
    '''句向量, 目前支持m3e, bge, simbert, text2vec-base-chinese
    :param checkpoint_path: str, 模型所在文件夹地址
    :param device: str, cpu/cuda
    :param model_config: dict, build_transformer_model时候用到的一些参数

    ```python
    >>> from bert4torch.pipelines import Text2Vec
    >>> sentences_1 = ["样例数据-1", "样例数据-2"]
    >>> sentences_2 = ["样例数据-3", "样例数据-4"]
    >>> text2vec = Text2Vec(checkpoint_path='bge-small-zh-v1.5', device='cuda')
    >>> embeddings_1 = text2vec.encode(sentences_1, normalize_embeddings=True)
    >>> embeddings_2 = text2vec.encode(sentences_2, normalize_embeddings=True)
    >>> similarity = embeddings_1 @ embeddings_2.T
    >>> print(similarity)
    ```
    '''
    def __init__(self, checkpoint_path:str, config_path:str=None, device:str=None, **kwargs) -> None:
        super().__init__(checkpoint_path, config_path=config_path, device=device, **kwargs)
        pooling = self.config.get('pooling', {})
        self.pool_strategy = self.config.get('pool_strategy', pooling.get('pool_strategy', 'cls'))  # 兼容老版本
        self.prompts = pooling.get('prompts', {})
        self.default_prompt_name = pooling.get('default_prompt_name')
        
    @torch.inference_mode()
    def encode(
        self,
        # sentence_transformers的encode方法参数
        sentences: Union[str, List[str]],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 8,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,

        # 独有参数
        truncate_dim:int = None,
        pool_strategy: PoolStrategy = None,
        custom_layer: Union[int, List[int]] = None,
        max_seq_length: int = None
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        '''
        :param sentences: Union[str, List[str]], 待编码文本
        :param prompt_name: str | None, 提示词的key
        :param prompt: str | None, 直接传入提示词
        :param batch_size: int, encode的batch_size
        :param show_progress_bar: bool, 是否展示编码的进度条
        :param convert_to_numpy: bool, 是否将输出转换为numpy
        :param convert_to_tensor: bool, 是否将输出转换为tensor
        :param normalize_embeddings: bool, 是否将输出的句向量进行归一化
        :param pool_strategy: str, 句向量的池化方式
        :param custom_layer: str, 自定义的句向量层
        :param max_seq_length: int, 最大序列长度

        :return: Union[List[torch.Tensor], np.ndarray, torch.Tensor]
        '''
        
        if convert_to_tensor:
            convert_to_numpy = False
    
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True
        
        # 获取提示词并对sentences进行预处理
        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                log_warn(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

        # 按照长度排序，避免padding过多
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        # 按照batch_size进行分批推理
        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            if self.tokenizer_type == 'b4t':
                batch_input = self.tokenizer(sentences_batch, max_length=max_seq_length, return_tensors='pt').to(self.device)
                output = self.model(batch_input)
            else:
                batch_input = self.tokenizer(sentences_batch, max_length=max_seq_length, return_tensors='pt', padding=True).to(self.device)
                output = self.model(**batch_input)

            # pooling
            last_hidden_state = output.get('last_hidden_state')
            pooler = output.get('pooled_output')
            if isinstance(batch_input, list):
                attention_mask = (batch_input[0] != self.tokenizer.pad_token_id).long()
            elif isinstance(batch_input, torch.Tensor):
                attention_mask = (batch_input != self.tokenizer.pad_token_id).long()
            else:  # 类似字典格式的
                attention_mask = batch_input.get('attention_mask', (batch_input['input_ids'] != self.tokenizer.pad_token_id).long())

            pool_strategy = pool_strategy or self.pool_strategy
            embs = get_pool_emb(last_hidden_state, pooler, attention_mask, pool_strategy, custom_layer)
            
            # 后处理
            if truncate_dim is not None:
                embs = embs[:, :truncate_dim]  # 截断维度
            if normalize_embeddings:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)  # 归一化
            if convert_to_numpy:
                embs = embs.cpu()  # 转为numpy
            all_embeddings.extend(embs)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings    