"""Bradley-Terry pairwise loss placeholder."""

import torch

def bradley_terry_loss(pred_a, pred_b, reduction='mean'):
    """Compute a simple Bradley-Terry style loss between two predictions."""
    # probability that a > b
    p = torch.sigmoid(pred_a - pred_b)
    loss = -torch.log(p + 1e-8)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss
