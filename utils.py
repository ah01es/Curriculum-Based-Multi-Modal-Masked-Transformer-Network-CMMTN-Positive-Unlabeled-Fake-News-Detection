import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
