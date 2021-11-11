from typing import OrderedDict
import numpy as np
import torch
import flwr as fl


def set_weights(self, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
    )
    self.load_state_dict(state_dict, strict=True)
