from collections import OrderedDict
import torch
from torch import nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class NeuralNetwork(nn.Module):
    def __init__(self,
                 chunk_size: int = 1280,
                 num_electrodes: int = 21,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 1,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(NeuralNetwork, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #Metrics
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