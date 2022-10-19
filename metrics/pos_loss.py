from typing import Optional

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


class CrossEntropyLossPerPosition(Module):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLossPerPosition, self).__init__()
        self.ce_loss = CrossEntropyLoss(weight, size_average, ignore_index, reduce,
                                        label_smoothing=label_smoothing,
                                        reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.stack([self.ce_loss(input[i], target[i])[1:-1] for i in range(input.shape[0])])
