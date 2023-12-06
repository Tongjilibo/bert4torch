from torch import nn
import torch
from typing import List, Optional, Union
import numpy as np
from bert4torch.snippets import torch_div


class CRF(nn.Module):
    '''Conditional random field: https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/layers/crf.py
    '''
    def __init__(self, num_tags: int, init_transitions: Optional[List[np.ndarray]] = None, freeze=False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        if (init_transitions is None) and (not freeze):
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
        elif init_transitions is not None:
            transitions = torch.tensor(init_transitions[0], dtype=torch.float)
            start_transitions = torch.tensor(init_transitions[1], dtype=torch.float)
            end_transitions = torch.tensor(init_transitions[2], dtype=torch.float)

            if not freeze:
                self.transitions = nn.Parameter(transitions)
                self.start_transitions = nn.Parameter(start_transitions)
                self.end_transitions = nn.Parameter(end_transitions)
            else:
                self.register_buffer('transitions', transitions)
                self.register_buffer('start_transitions', start_transitions)
                self.register_buffer('end_transitions', end_transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, mask: torch.ByteTensor,
                tags: torch.LongTensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            emissions: [btz, seq_len, num_tags]
            mask: [btz, seq_len]
            tags: [btz, seq_len]
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm."""
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        best_path = self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)
        return best_path[0] if nbest == 1 else best_path

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, '
                             f'got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError('the first two dimensions of emissions and tags must match, '
                                 f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError('the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq_bf = mask[:, 0].all()
            if not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)
        # tags: (batch_size, seq_length)
        # mask: (batch_size, seq_length)
        batch_size, seq_length = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1
        # shape: (batch_size,)
        last_tags = tags[torch.arange(batch_size), seq_ends]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)
        # mask: (batch_size, seq_length)
        seq_length = emissions.size(1)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
                              nbest: int, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (batch_size, seq_length, num_tags)
        # mask: (batch_size, seq_length)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        batch_size, seq_length = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[:, 0]
        history_idx = torch.zeros((batch_size, seq_length, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((batch_size, seq_length, nbest), pad_tag, dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[:, i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((batch_size, seq_length, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[:, idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[:, idx] = torch_div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')  # 兼容老版本
            # best_tags_arr[:, idx] = torch.div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')

        return torch.where(mask.unsqueeze(-1).bool(), best_tags_arr, oor_tag).permute(2, 0, 1)
