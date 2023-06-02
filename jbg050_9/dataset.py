import torch
from torch_geometric.data import InMemoryDataset, Data


class CrimeDataset(InMemoryDataset):
    """
    torch-geometric InMemoryDataset class
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(CrimeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graph_data.pt']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        pass

    def process(self):
        data = torch.load(self.raw_paths[0])
        data_list = []
        for t in range(1, data.x.size(1)):
            x_t = data.x[:, t-1, :]  # Node features at previous timestep
            y_t = data.y[:, t]  # 'Burglary' to predict at current timestep
            edge_index_t = data.edge_index[:, data.edge_index[1] < t]  # only keep edges up to current timestep
            edge_attr_t = data.edge_attr[data.edge_index[1] < t]
            data_t = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
            data_list.append(data_t)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CrimeDatasetSW(InMemoryDataset):
    def __init__(self, root, sequence_length=24, transform=None, pre_transform=None):
        super(CrimeDatasetSW, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.sequence_length = sequence_length

    @property
    def raw_file_names(self):
        return ['graph_data_sw.pt']

    @property
    def processed_file_names(self):
        return ['processed_data_sw.pt']

    def download(self):
        pass

    def process(self):
        data = torch.load(self.raw_paths[0])
        data_list = []
        sequence_length = self.sequence_length  # define your sequence length
        for t in range(sequence_length, data.x.size(1)):
            x_sequence = data.x[:, t-sequence_length:t, :]  # Node features for the sequence
            y_t = data.y[:, t]  # 'Burglary' to predict at next timestep
            edge_index_sequence = [data.edge_index[:, data.edge_index[1] < t] for _ in range(sequence_length)]  # Edges for each timestep in the sequence
            edge_attr_sequence = [data.edge_attr[data.edge_index[1] < t] for _ in range(sequence_length)]  # Edge attributes for each timestep in the sequence
            data_t = Data(x=x_sequence, edge_index=edge_index_sequence, edge_attr=edge_attr_sequence, y=y_t)
            data_list.append(data_t)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
