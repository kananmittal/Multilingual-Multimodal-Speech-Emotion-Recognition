import torch
import torch.nn as nn


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.prototypes

    def prototype_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """
        Pull embeddings towards their class prototype and push from others.
        embeddings: [B, D], labels: [B]
        """
        # Pull to own prototype
        pos = torch.norm(embeddings - self.prototypes[labels], dim=1).mean()
        # Push from other prototypes using log-sum-exp over negative distances
        dists = torch.cdist(embeddings, self.prototypes)  # [B, C]
        arange = torch.arange(dists.size(0), device=dists.device)
        pos_mask = torch.zeros_like(dists).bool()
        pos_mask[arange, labels] = True
        neg_dists = dists.masked_fill(pos_mask, float('inf'))  # mask own class
        # Use soft-min over negatives via -logsumexp(-d)
        neg = -torch.logsumexp(-neg_dists, dim=1)
        neg = neg.mean()
        return pos + margin - neg


