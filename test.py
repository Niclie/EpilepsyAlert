from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.dataset import load_dataset
from src.model.train import NeuralNetwork, train, test
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#ids = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22']
batch_size = 32
local_epochs = 5
#for id in ids:
id = 'chb20'
# for _ in range(10):
print(id)
client_model = NeuralNetwork().to(DEVICE)
#client_model.load_state_dict(torch.load("generic_model_weights.pth"))

dataset = load_dataset(id)
train_dataset = TensorDataset(torch.tensor(dataset['train_data'], dtype=torch.float32),
                              torch.tensor(dataset['train_labels'], dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.tensor(dataset['test_data'], dtype=torch.float32),
                             torch.tensor(dataset['test_labels'], dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loss = train(client_model, train_loader, local_epochs, DEVICE, verbose=True)
# print(f'train_loss: {train_loss}')
res = test(client_model, test_loader, DEVICE)
print(f'Loss: {res[0]:.2f}, Accuracy: {res[1]:.2f}, Precision: {res[2]:.2f}')
print()