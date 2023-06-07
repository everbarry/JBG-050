import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, GATConv
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
#
from torch_geometric_temporal.nn import ASTGCN



class TimeSeriesNodeDataset(Dataset):
    def __init__(self, data, sequence_length=5):
        super(TimeSeriesNodeDataset, self).__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # We subtract the sequence length because for each sequence of "sequence_length",
        # we use it to predict the next step, so we will have fewer total examples by "sequence_length"
        return self.data.shape[1] - self.sequence_length

    def __getitem__(self, index):
        # We use "index:index+sequence_length" to get the inputs
        # and then "index+sequence_length" to get the corresponding target
        x = self.data[:, index:index+self.sequence_length]
        y = self.data[:, index+self.sequence_length]
        return x, y




data = torch.load('../data/raw/graph_data.pt')
device = 'cuda'
edge_index = data.edge_index.to(device)
edge_weight = data.edge_attr.to(device)
data = data.y.to(device)
print(f'data starting shape: {data.shape}')

split_index = int(data.shape[1] * 0.8)

# Create training and test datasets
train_data = data[:, :split_index]
test_data = data[:, split_index:-5] # We also need to consider the sequence_length for the test data

train_dataset = TimeSeriesNodeDataset(train_data, sequence_length=12)
test_dataset = TimeSeriesNodeDataset(test_data, sequence_length=12)

# Create DataLoaders. Since we want all nodes at once, we set batch_size to the number of nodes.
# Since the data is already on the desired device (CPU or GPU), we set pin_memory to False.
train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=False)


class Monster(nn.Module):
    def __init__(self, node_features=1):
        super(Monster, self).__init__()
        self.att_fn = nn.LeakyReLU(negative_slope=0.01)
        self.gat = ASTGCN(nb_blocks=1, in_channels=1, K=2)

    def forward(self, x, edge_index, edge_weight):
        h = self.gat(x, edge_index)
        return h

model = ASTGCN(nb_block=2, in_channels=1, K=2, nb_chev_filter=2, nb_time_filter=6, time_strides=2, num_for_predict=1, len_input=12, num_of_vertices=data.shape[0])
model.to(device)

optimizer = Adam(model.parameters(), lr=0.005, weight_decay=0.001, amsgrad=True)
criterion = nn.MSELoss()


losses, r2_scores = [], []
for epoch in tqdm(range(100), desc='Training...'):
    epoch_losses, epoch_r2_scores = [], []
    for x, y in tqdm(train_loader, desc=f'batch {epoch}'):
        x, y = x.to(device), y.squeeze(0).to(device)
        #x, y = x.squeeze(0).to(device), y.squeeze(0).to(device)
        optimizer.zero_grad()
        H = model(x.view(1, x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        loss = criterion(H, y)
        #loss = criterion(H.squeeze(1), y)
        epoch_losses.append(loss.item())
        epoch_r2_scores.append(r2_score(y.cpu().numpy(), H.detach().cpu().numpy()))
        loss.backward()
        optimizer.step()
    losses.append(np.mean(epoch_losses))
    r2_scores.append(np.mean(epoch_r2_scores))
    tqdm.write(f"{'-'*20}\nEpoch : {epoch}\nLoss (MSE) = {losses[-1]:.4f}\nR2 Score = {max(r2_scores):.4f}")
