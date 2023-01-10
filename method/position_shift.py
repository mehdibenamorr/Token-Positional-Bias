import numpy as np
import torch


def random_shift(batch, max_length=None, k=1):
    pos_ids = [i for i in range(max_length)]
    position_ids = []
    for input_ids in list(batch["input_ids"]):
        indices = input_ids.nonzero() # inputs_ids.nonzero() -> tensor
        # Random position k
        k = np.random.choice(pos_ids[len(indices):-len(indices)])
        indices = indices[1:]
        shifted_pos = np.array(pos_ids)
        shifted_pos[indices] += k
        position_ids.append(shifted_pos.tolist())
    batch["position_ids"] = torch.Tensor(position_ids).to(dtype=batch["input_ids"].dtype)

    return batch
