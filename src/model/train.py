from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=64, kernel_size=16, padding='same') #padding='same'
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=8, padding='same') #padding='same'
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(32, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.global_avg_pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)

        x = F.relu(self.fc2(x))
        x = self.dropout4(x)

        x = torch.sigmoid(self.fc3(x))

        return x


def train(model, dataloader, epochs, device, verbose=True):
    loss_fn = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    running_loss = 0.0
    for t in range(epochs):
        correct, epoch_loss = 0, 0.0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X).squeeze()
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            predicted = (output.data >= 0.5).float()
            correct += (predicted == y).sum().item()

        epoch_loss /= len(dataloader)
        epoch_acc = correct / len(dataloader.dataset)
        if verbose:
            print(f'Epoch {t+1}: Loss {epoch_loss:.2f}, Accuracy {epoch_acc:.2f}')

    avg_trainloss = running_loss / len(dataloader)

    return avg_trainloss


def test(model, dataloader, device):
    loss_fn = nn.BCELoss().to(device)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X).squeeze()
            test_loss += loss_fn(output, y).item()
            predicted = (output.data >= 0.5).float()
            correct += (predicted == y).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)

    return test_loss, correct


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: NeuralNetwork, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)