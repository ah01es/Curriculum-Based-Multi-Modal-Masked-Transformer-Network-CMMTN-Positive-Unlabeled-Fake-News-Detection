import torch
import torch.nn.functional as F

def nnpu_risk(logits_p, logits_u, prior, reduction="mean"):
    prior = float(prior)
    y_pos = torch.ones_like(logits_p)
    y_neg_p = torch.zeros_like(logits_p)
    y_neg_u = torch.zeros_like(logits_u)

    R_p_pos = F.binary_cross_entropy_with_logits(logits_p, y_pos, reduction=reduction)
    R_p_neg = F.binary_cross_entropy_with_logits(logits_p, y_neg_p, reduction=reduction)
    R_u_neg = F.binary_cross_entropy_with_logits(logits_u, y_neg_u, reduction=reduction)

    term = R_u_neg - prior * R_p_neg
    if reduction == "mean":
        term = torch.clamp(term, min=0.0)
        risk = prior * R_p_pos + term
    else:
        term = torch.clamp(term, min=0.0)
        risk = prior * R_p_pos + term
    return risk
