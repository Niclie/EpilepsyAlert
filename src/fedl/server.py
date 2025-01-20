from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from src.model.train import get_weights, NeuralNetwork


class AggregateCustomMetricStrategy(FedAvg):
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
        print(
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}


def server_fn(context: Context):
    num_rounds = 10
    arrays = get_weights(NeuralNetwork())
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