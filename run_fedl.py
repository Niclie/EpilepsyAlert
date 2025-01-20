from flwr.simulation import run_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp
from src.fedl.client import client_fn
from src.fedl.server import server_fn


def start_simulation():
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1}}

    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=7, #7
        backend_config=backend_config
    )

start_simulation()