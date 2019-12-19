import torch
from torch.testing import get_all_dtypes

dtypes = get_all_dtypes()
dtypes.remove(torch.half)
dtypes.remove(torch.bool)
if hasattr(torch, 'bfloat16'):
    dtypes.remove(torch.bfloat16)

grad_dtypes = [torch.float, torch.double]

devices = [torch.device('cpu')]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
