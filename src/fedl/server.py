from collections import OrderedDict
from typing import Union, Optional
import flwr as fl
import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters, FitRes, Parameters, Scalar
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from src.model.train import get_weights, NeuralNetwork

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetwork().to(DEVICE)


class AggregateCustomMetricStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(net.state_dict(), f"Hospital_C/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        # print(
        #     f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        # )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}


def server_fn(context: Context):
    num_rounds = 20

    model = NeuralNetwork()
    #model.load_state_dict(torch.load("generic_model_weights.pth"))
    arrays = get_weights(model)
    parameters = ndarrays_to_parameters(arrays)

    # strategy = FedAvg(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1,
    #     initial_parameters=parameters
    # )

    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=parameters
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)