"""
参考transformers.generation.logits_process.py
url: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

- 做了部分简化，没有区分LogitsProcessor和LogitsWarper
"""
import torch
from bert4torch.snippets import add_start_docstrings, log_warn_once
from typing import Union


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            scores = processor(input_ids, scores, **kwargs)
        return scores
    

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """重复性惩罚
    """
    def __init__(self, penalty: float):
        if not isinstance(penalty, (int, float)) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.penalty == 1.0:
            return scores
        
        # 拼接历史的输入input_ids
        states: dict = kwargs.get('states')
        output_ids: torch.LongTensor = kwargs.get('output_ids')
        if (states is not None) and (states.get('past_token_ids') is not None):
            past_token_ids = states['past_token_ids']
        elif len(input_ids.shape) == len(output_ids.shape):
            past_token_ids = torch.cat([input_ids, output_ids], 1)
        else:
            past_token_ids = output_ids
    
        score = torch.gather(scores, 1, past_token_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, past_token_ids, score)
        return scores_processed


class TopPLogitsWarper(LogitsProcessor):
    """TopP采样
    """
    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if top_p is not None and (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.top_p is None:
            return scores
        
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TopKLogitsWarper(LogitsProcessor):
    """TopK采样
    """
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.top_k is None:
            return scores
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
    

class TemperatureLogitsWarper(LogitsProcessor):
    """Temperature
    """
    def __init__(self, temperature: Union[float, int]):
        if not isinstance(temperature, (float, int)) or temperature < 0:
            raise ValueError(f"`temperature` (={temperature}) has to be a strictly positive or zero float")
        if isinstance(temperature, (float, int)) and temperature == 0.0:
            log_warn_once("`temperature` == 0 means greedy decoding strategies")
        self.temperature = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.temperature == 1 or self.temperature == 0:
            return scores
        scores_processed = scores / self.temperature
        return scores_processed