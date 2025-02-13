import torch


class Round:
    def __call__(self, mask):
        return torch.round(mask)
