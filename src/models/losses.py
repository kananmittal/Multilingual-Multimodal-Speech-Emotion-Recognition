import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta: float = 0.9999, gamma: float = 2.0, num_classes: int | None = None):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.register_buffer("effective_num", torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute class weights using effective number if num_classes provided
        if self.num_classes is not None:
            with torch.no_grad():
                counts = torch.bincount(targets, minlength=self.num_classes).float().clamp(min=1.0)
                effective_num = 1.0 - torch.pow(torch.tensor(self.beta, device=logits.device), counts)
                weights = (1.0 - self.beta) / effective_num
                weights = weights / weights.sum() * self.num_classes
        else:
            weights = None

        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        ce = F.cross_entropy(logits, targets, reduction='none', weight=weights)
        loss = (focal_weight * ce).mean()
        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D]
        labels: [B]
        """
        features = F.normalize(features, dim=-1)
        logits = features @ features.t() / self.temperature  # [B, B]
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        labels = labels.contiguous()
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits_mask = 1.0 - torch.eye(features.size(0), device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


