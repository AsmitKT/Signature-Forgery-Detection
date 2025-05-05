import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Computes triplet loss:
      L = mean( max(||A–P||₂ – ||A–N||₂ + margin, 0) )
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                anchor:   torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses   = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()
