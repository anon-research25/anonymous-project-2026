from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedVeracityLoss(nn.Module):
    """L_sup = -sum_i y_i log(pi_i), computed stably from graph logits.

    No reward, explanation loss, agent fine-tuning or differentiable LLM verdict
    path is introduced: the manuscript specifies graph cross entropy only.
    """

    def forward(self, logits, targets):
        if targets.ndim == logits.ndim:
            if targets.shape != logits.shape:
                raise ValueError("One-hot targets must have the same shape as logits")
            return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        return F.cross_entropy(logits, targets.long())


class RouterClassificationLoss(nn.Module):
    """Cross entropy for the S/M/C language-model router."""

    def forward(self, logits, strategy_ids):
        return F.cross_entropy(logits, strategy_ids.long())
