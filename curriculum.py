import torch

def select_confident(logits_u, indices_u, step_pos, step_neg):
    probs = torch.sigmoid(logits_u).detach().cpu()
    pos_sorted = torch.argsort(probs, descending=True)
    neg_sorted = torch.argsort(probs, descending=False)
    sel_pos = pos_sorted[:step_pos]
    sel_neg = neg_sorted[:step_neg]
    return indices_u[sel_pos].tolist(), indices_u[sel_neg].tolist()
