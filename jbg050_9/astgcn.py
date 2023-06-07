import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric_temporal.nn import ASTGCN, MTGNN, TGCN2
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt



data = torch.load('../data/raw/graph_data.pt')
device = 'cuda'
#data = data.to(device)
edge_index = data.edge_index.to(device)
edge_weight = data.edge_attr.to(device)

print(f'data starting shape:\nx = {data.x.shape}\ny = {data.y.shape}')

x = data.x.unsqueeze(0)
y = data.y.unsqueeze(0)
y = data.y.unsqueeze(-1)
print(f'x: {x.shape}, y: {y.shape}')

split_index = int(x.size(2) * 0.9)  # 90% of the time steps
x_train = x[:, :, :split_index, :]
x_test = x[:, :, split_index:, :]
y_train = y[:, :, :split_index]
y_test = y[:, :, split_index:]
print(f'x train: {x_train.shape}, test: {x_test.shape}\ny train: {y_train.shape}, test: {y_test.shape}')


class ASTGCNDataset(Dataset):
    def __init__(self, X, y, lookback, forecast):
        self.X = X.squeeze(1)  # remove dimensions of size 1
        self.y = y
        self.lookback = lookback
        self.forecast = forecast

    def __len__(self):
        # Adjust for the lookback and forecast
        return self.X.shape[2] - self.lookback - self.forecast + 1

    def __getitem__(self, idx):
        # Get the data and label for a window
        X = self.X[:, :, idx : idx+self.lookback]
        y = self.y[:, idx+self.lookback : idx+self.lookback+self.forecast]

        return X, y




lookback = 12
forecast = 1
train_dataset = ASTGCNDataset(x_train, y_train, lookback, forecast)
test_dataset = ASTGCNDataset(x_test, y_test, lookback, forecast)

# Create DataLoader objects
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = ASTGCN(nb_block=1, in_channels=16, K=2, nb_chev_filter=2, nb_time_filter=6, time_strides=2, num_for_predict=forecast, len_input=lookback, num_of_vertices=13692)
#model = TGCN2(in_channels=16, out_channels=1, improved=True, add_self_loops=True, batch_size=batch_size)

model.to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss() # Modify based on your specific needs

num_epochs = 30
# Training
losses = []
for epoch in tqdm(range(num_epochs)):
    for X, y in train_loader:
        torch.cuda.empty_cache()
        X = X.squeeze(1)
        X = X.permute(0, 1, 3, 2)  # rearrange dimensions to (batch_size, num_nodes, num_features, num_timesteps)
        tqdm.write(f"Shape of X: {X.shape}")
        optimizer.zero_grad()
        pred = model(X.to(device), edge_index)

        loss = loss_fn(pred, y)tr
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

# Testing
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for X, y in test_loader:
        pred = model(X, edge_index)

        y_true.append(y.numpy())
        y_pred.append(pred.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss over time')

plt.subplot(1, 2, 2)
plt.bar(['R2 Score'], [r2])
plt.title('R2 Score')

plt.show()
