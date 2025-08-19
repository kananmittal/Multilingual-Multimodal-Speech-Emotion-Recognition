import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def weighted_f1(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, preds, average='weighted')


def energy_score(logits):
    # For OOD detection we use negative log-sum-exp
    return -torch.logsumexp(logits, dim=1)
