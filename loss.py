import torch
import torch.nn as nn


class SigmoidLoss(nn.Module):
    """Binary sigmoid loss for outputs with two logits per sample.

    This computes a single logit as (logit_pos - logit_neg) and applies
    BCEWithLogitsLoss against binary targets (0/1).
    """
    def __init__(self, weight=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, outputs, targets):
        # outputs: (N, 2) logits; targets: (N,) with 0/1
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        if outputs.size(1) == 2:
            logits = outputs[:, 1] - outputs[:, 0]
        else:
            # assume single-logit output
            logits = outputs.squeeze(1)

        targets = targets.float()
        return self.criterion(logits, targets)
