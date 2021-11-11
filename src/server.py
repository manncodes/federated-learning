from typing import List, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import EvaluateRes

import cifar
from util import set_weights

# from client import DEVICE
from cifar import DATA_ROOT
from run_all import NUM_CLIENTS
from config import *

import wandb

run_name = f"server-{datetime.now()}"

DATA_ROOT = "../data"


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round_weights/round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        wandb.log({"round": rnd, "server_aggregated_accuracy": accuracy_aggregated})
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def get_eval_fn(
    testset: CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = cifar.load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        testset = torch.utils.data.subset(CIFAR10, range(1000))
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return cifar.test(model, testloader, device=DEVICE)

    return evaluate


def run_default_stratergy():
    config = {
        "name": "cifar10_flwr_basic",
        "num_clients": NUM_CLIENTS,
        "rounds": NUM_ROUNDS,
        "epochs": 1,
        "batch_size": 32,
        "test_set_N": 1000,
        "strategy": fl.server.strategy.FedAvg,
        "strat_param": {"fraction_fit": FRACTION_FIT, "fraction_eval": FRACTION_EVAL,},
    }

    wandb.init(project="cifar-flwr-basic", name=run_name)

    # Define strategy
    """ 
    FedAdam 
    -------    
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    """
    # strategy = fl.server.strategy.FedAdam(fraction_fit=0.5, fraction_eval=0.5,inital_parameters)

    strategy = fl.server.strategy.FedAvg(fraction_fit=0.5, fraction_eval=0.5)

    # Configure logger and start server
    fl.common.logger.configure("server")

    # Start server
    fl.server.start_server(
        server_address="127.0.0.1:8080", config={"num_rounds": 10}, strategy=strategy,
    )


def run_custom_stratergy():
    config = {
        "name": "cifar10_flwr",
        "num_clients": NUM_CLIENTS,
        "rounds": NUM_ROUNDS,
        "epochs": 1,
        "batch_size": 32,
        "test_set_N": 1000,
        "strategy": AggregateCustomMetricStrategy,
        "strat_param": {"fraction_fit": FRACTION_FIT, "fraction_eval": FRACTION_EVAL,},
    }

    wandb.init(project="cifar-flwr", name=run_name)

    # Define strategy
    strategy = AggregateCustomMetricStrategy(fraction_fit=0.5, fraction_eval=0.5,)

    # Configure logger and start server
    fl.common.logger.configure("server")

    # Start server
    fl.server.start_server(
        server_address="127.0.0.1:8080", config={"num_rounds": 3}, strategy=strategy,
    )


if __name__ == "__main__":
    run_default_stratergy()

    # run_custom_stratergy()
