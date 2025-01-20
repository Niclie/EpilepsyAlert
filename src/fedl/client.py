import torch
from torch.utils.data import TensorDataset, DataLoader
from preprocessing.dataset import load_dataset
from src.model.train import get_weights, set_weights, train, test, NeuralNetwork
from flwr.client import NumPyClient


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {'loss': loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}



HOSPITAL = {0: 'chb01', 1: 'chb02', 2: 'chb03', 3: 'chb04', 4: 'chb05', 5: 'chb06', 6: 'chb07'} # HOSPITAL_A
# HOSPITAL = {0: 'chb08', 1: 'chb09', 2: 'chb10', 3: 'chb11', 4: 'chb14', 5: 'chb15'} # HOSPITAL_B
# HOSPITAL = {0: 'chb16', 1: 'chb17', 2: 'chb18', 3: 'chb19', 4: 'chb20', 5: 'chb21', 6: 'chb22'} # HOSPITAL_C

def client_fn(context):
    patient_id = HOSPITAL.get(context.node_config['partition-id'])
    print(patient_id)
    dataset = load_dataset(patient_id)
    #dataset = get_preprocessed_dataset(patient_id)
    #print(dataset['train_data'].shape)
    batch_size = 32 #64
    local_epochs = 5

    ###################
    # train_data = dataset['train_data'].values  # Converte DataFrame in array NumPy
    # test_data = dataset['test_data'].values
    # train_labels = dataset['train_labels']  # Supponiamo sia già un array NumPy
    # test_labels = dataset['test_labels']
    #
    # train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
    #                           torch.tensor(train_labels, dtype=torch.float32))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    # test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
    #                              torch.tensor(test_labels, dtype=torch.float32))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    ###################

    train_dataset = TensorDataset(torch.tensor(dataset['train_data'], dtype=torch.float32),
                              torch.tensor(dataset['train_labels'], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(dataset['test_data'], dtype=torch.float32),
                                 torch.tensor(dataset['test_labels'], dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    net = NeuralNetwork()

    return FlowerClient(net, train_loader, test_loader, local_epochs).to_client()