"""Flower client example using PyTorch for CIFAR-10 image classification."""

import os
import sys
import timeit
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common.typing import Config, PropertiesRes
import numpy as np
import torch
import wandb


run_name = f"client-id-{sys.argv[1]}-{os.getpid()}-{datetime.now()}"
wandb.init(project="cifar-flwr", name=run_name)

import cifar

USE_FEDBN: bool = True

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
print(f"{DEVICE = }")

# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name  # dont send bn params
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        wandb.log({"client_test_loss": loss, "client_test_accuracy": accuracy})
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

    def get_properties(self, config: Config) -> PropertiesRes:
        wandb.log({"client_config": config})
        return super().get_properties(config)


def main() -> None:
    """Load data, start CifarClient."""

    # Load data
    trainloader, testloader = cifar.load_data()

    # Load model
    model = cifar.Net().to(DEVICE).train()
    wandb.watch(model)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = CifarClient(model, trainloader, testloader)
    fl.client.start_numpy_client("127.0.0.1:8080", client)


if __name__ == "__main__":
    main()
