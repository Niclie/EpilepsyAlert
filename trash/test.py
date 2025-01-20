import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from src.preprocessing.dataset import load_dataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=64, kernel_size=16) #padding='same'
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=8) #, padding='same'
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(32, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
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


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X).squeeze()
            test_loss += loss_fn(output, y).item()
            predicted = (output >= 0.5).float()
            correct += (predicted == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# device = ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using {device} device')
#
# dataset = load_dataset('chb06')
# batch_size = 64
# train_dataset = TensorDataset(torch.tensor(dataset['train_data'], dtype=torch.float32),
#                               torch.tensor(dataset['train_labels'], dtype=torch.float32))
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = TensorDataset(torch.tensor(dataset['test_data'], dtype=torch.float32),
#                              torch.tensor(dataset['test_labels'], dtype=torch.float32))
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
# model = NeuralNetwork().to(device)
# loss_fn = nn.BCELoss().to(device)
# optimizer = torch.optim.Adam(model.parameters())
#
# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_loader, model, loss_fn, optimizer, device)
#     test_loop(test_loader, model, loss_fn)
# print("Done!")

import src.model.train
