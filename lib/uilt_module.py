import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class CurveLoss(nn.MSELoss):
    def __init__(self, model, lambda_l1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.model = model
        self.lambda_l1 = lambda_l1

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1_penalty = 0
        for param in self.model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        return super().forward(input, target) + torch.square(torch.mean(torch.abs(input - target))) + self.lambda_l1 * l1_penalty
