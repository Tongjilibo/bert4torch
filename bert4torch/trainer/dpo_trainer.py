from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import nn
from torch4keras.trainer import Trainer
import copy
import torch
import torch.nn.functional as F


class DPOTrainer(Trainer):
    def __init__(self, model, ref_model=None, pad_token_id=0, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        self.pad_token_id = pad_token_id
        self.criterion = DPOLoss(beta)

    def _get_batch_logps(self, logits, labels, average_log_prob):
        """Compute the log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(self, input_ids, input_labels):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        all_logits = self.model(input_ids).to(torch.float32)
        all_logps = self._get_batch_logps(all_logits, input_labels, average_log_prob=False)
        split = input_ids.shape[0] // 2
        chosen_logps = all_logps[:split]
        rejected_logps = all_logps[split:]

        chosen_logits = all_logits[:split]
        rejected_logits = all_logits[split:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def forward(self, input_ids, input_labels):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = self.concatenated_forward(model, input_ids, input_labels)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _, _, = self.concatenated_forward(self.ref_model, input_ids)

        losses, chosen_rewards, rejected_rewards = self.criterion(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean()

class DPOLoss:
    def __init__(self, beta=0.1) -> None:
        self.beta = beta
    def __call__(self, policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
