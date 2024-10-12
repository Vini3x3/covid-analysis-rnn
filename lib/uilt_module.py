import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class CurveLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target) + torch.square(torch.mean(torch.abs(input - target)))
