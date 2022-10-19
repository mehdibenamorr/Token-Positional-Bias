import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossPerPosition(CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLossPerPosition, self).__init__(reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.stack([self.super()(input[i], target[i])[1:-1] for i in range(input.shape[0])])
