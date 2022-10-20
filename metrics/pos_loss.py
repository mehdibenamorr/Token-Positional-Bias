from typing import Optional, List, Union

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F


class CrossEntropyLossPerPosition(Module):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLossPerPosition, self).__init__()
        self.ce_loss = CrossEntropyLoss(weight, size_average, ignore_index, reduce,
                                        label_smoothing=label_smoothing,
                                        reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.stack([self.ce_loss(input[i], target[i])[1:-1] for i in range(input.shape[0])])


def padded_stack(
        tensors: List[torch.Tensor], side: str = "right", mode: str = "constant", value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out
