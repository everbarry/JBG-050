import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.nn import GCNConv, GATConv, ChebConv


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class GCNGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(GCNGCN, self).__init__()
        self.gcnconv = GCNConv(node_features, filters)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.gcnconv(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class ChebyshevGCN(torch.nn.Module):
    def __init__(self, node_features, filters, K):
        super(ChebyshevGCN, self).__init__()
        self.chebconv = ChebConv(node_features, filters, K)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.chebconv(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class AttentionGCN(torch.nn.Module):
    def __init__(self, node_features, filters, heads):
        super(AttentionGCN, self).__init__()
        self.gat = GATConv(node_features, filters, heads=heads)
        self.linear = torch.nn.Linear(filters*heads, 1)



    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

class AttentionGCNl(torch.nn.Module):
    def __init__(self, node_features, filters, heads):
        super(AttentionGCNl, self).__init__()
        self.gat1 = GATConv(node_features, filters, heads=heads, dropout=0.4)
        self.gat2 = GATConv(filters*heads, filters, heads=heads, dropout=0.4)
        self.linear1 = torch.nn.Linear(filters*heads, filters)
        self.linear2 = torch.nn.Linear(filters, 1)
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class AttentionGCNSW(torch.nn.Module):
    def __init__(self, node_features, filters, heads):
        super(AttentionGCNSW, self).__init__()
        self.gat = GATConv(node_features, filters, heads=heads)
        self.rnn = torch.nn.GRU(input_size=filters*heads, hidden_size=filters*heads, batch_first=True)
        self.linear = torch.nn.Linear(filters*heads, 1)

    def forward(self, x_sequence, edge_index_sequence):
        # Process each graph in the sequence with the GAT layer
        gat_outputs = []
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :]
            edge_index_t = edge_index_sequence[t]
            h_t = self.gat(x_t, edge_index_t)
            h_t = F.relu(h_t)
            gat_outputs.append(h_t)

        # Stack the GAT outputs into a tensor
        gat_outputs = torch.stack(gat_outputs, dim=1)

        # Process the sequence of graph embeddings with the RNN
        rnn_output, _ = self.rnn(gat_outputs)

        # Use the last RNN output to predict the target
        h = self.linear(rnn_output[:, -1, :])
        return h.squeeze(-1)

class DeeperAttentionGCNSW(torch.nn.Module):
    def __init__(self, node_features, filters1, heads1, filters2, heads2):
        super(DeeperAttentionGCNSW, self).__init__()
        self.gat1 = GATConv(node_features, filters1, heads=heads1, concat=True)
        self.gat2 = GATConv(filters1 * heads1, filters2, heads=heads2, concat=False)  # adjust this line
        self.rnn = torch.nn.GRU(input_size=filters2, hidden_size=filters2, batch_first=True)  # adjust this line
        self.linear = torch.nn.Linear(filters2, 1)

    def forward(self, x_sequence, edge_index_sequence):
        # Process each graph in the sequence with the GAT layers
        gat_outputs = []
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :]
            edge_index_t = edge_index_sequence[t]
            h_t = F.relu(self.gat1(x_t, edge_index_t))
            h_t = F.relu(self.gat2(h_t, edge_index_t))
            gat_outputs.append(h_t)

        # Stack the GAT outputs into a tensor
        gat_outputs = torch.stack(gat_outputs, dim=1)

        # Process the sequence of graph embeddings with the RNN
        rnn_output, _ = self.rnn(gat_outputs)

        # Use the last RNN output to predict the target
        h = self.linear(rnn_output[:, -1, :])
        return h.squeeze(-1)
