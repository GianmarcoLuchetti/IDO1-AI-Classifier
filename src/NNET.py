import torch
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool


class GIN_0(torch.nn.Module):
    def __init__(self):
        super(GIN_0, self).__init__()
        dim_h_conv = 15  # Hidden neurons conv layers
        dim_h_fc = dim_h_conv*5  # Hidden neurons MLP

        # GIN convolutional layers
        self.conv1 = GINConv(Sequential(Linear(16, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv5 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))

        # Fully connected layers
        self.lin1 = Linear(dim_h_fc, dim_h_fc)
        self.lin2 = Linear(dim_h_fc, 2)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h5 = self.conv5(h4, edge_index)

        # Graph level readout, pooling function
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # MLP classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.lin2(h)
        h = F.log_softmax(h, dim=1)

        return h


class GIN_1(torch.nn.Module):
    def __init__(self):
        super(GIN_1, self).__init__()
        dim_h_conv = 35  # Hidden neurons conv layers
        dim_h_fc = dim_h_conv*5  # Hidden neurons MLP

        # GIN convolutional layers
        self.conv1 = GINConv(Sequential(Linear(16, dim_h_conv), # 14
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv5 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))

        # Fully connected layers
        self.lin1 = Linear(dim_h_fc, dim_h_fc)
        self.lin2 = Linear(dim_h_fc, 4)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h4 = F.dropout(h4, p=0.1, training=self.training)
        h5 = self.conv5(h4, edge_index)

        # Graph level readout, pooling function
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # MLP classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.lin2(h)
        h = F.log_softmax(h, dim=1)

        return h


class GIN_2(torch.nn.Module):
    def __init__(self):
        super(GIN_2, self).__init__()
        dim_h_conv = 50  # Hidden neurons conv layers
        dim_h_fc = dim_h_conv*5  # Hidden neurons MLP

        # GIN convolutional layers
        self.conv1 = GINConv(Sequential(Linear(16, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv5 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))

        # Fully connected layers
        self.lin1 = Linear(dim_h_fc, dim_h_fc)
        self.lin2 = Linear(dim_h_fc, 4)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h5 = self.conv5(h4, edge_index)

        # Graph level readout, pooling function
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # MLP classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        h = F.log_softmax(h, dim=1)

        return h


class GIN_3(torch.nn.Module):
    def __init__(self):
        super(GIN_3, self).__init__()
        dim_h_conv = 25  # Hidden neurons conv layers
        dim_h_fc = dim_h_conv*5  # Hidden neurons MLP

        # GIN convolutional layers
        self.conv1 = GINConv(Sequential(Linear(16, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))
        self.conv5 = GINConv(Sequential(Linear(dim_h_conv, dim_h_conv),
                                        BatchNorm1d(dim_h_conv), ReLU(),
                                        Linear(dim_h_conv, dim_h_conv), ReLU()))

        # Fully connected layers
        self.lin1 = Linear(dim_h_fc, dim_h_fc)
        self.lin2 = Linear(dim_h_fc, 4)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h4 = F.dropout(h4, p=0.15, training=self.training)
        h5 = self.conv5(h4, edge_index)

        # Graph level readout, pooling function
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # MLP classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.lin2(h)
        h = F.log_softmax(h, dim=1)

        return h


class GCN_1(torch.nn.Module):
    def __init__(self):
        super(GCN_1, self).__init__()
        dim_h = 60  # Hidden neurons conv layers
        # Convolutional layers
        self.conv1 = GCNConv(16, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)

        # Fully connected layers
        self.lin = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, dim_h)
        self.lin3 = Linear(dim_h, 4)

        # Batch normalization layers
        self.bn1 = BatchNorm1d(dim_h)
        self.bn2 = BatchNorm1d(dim_h)
        self.bn3 = BatchNorm1d(dim_h)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h = self.conv1(x, edge_index)
        h = h.relu()

        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=0.05, training=self.training)

        h = F.relu(self.conv3(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=0.1, training=self.training)

        # Graph-level readout, pooling function
        hG = global_mean_pool(h, batch)

        # MLP classifier
        hG = F.relu(self.lin(hG))
        hG = self.bn3(hG)
        hG = F.relu(self.lin2(hG))
        hG = F.dropout(hG, p=0.35, training=self.training)
        hG = self.lin3(hG)

        return F.log_softmax(hG, dim=1)


class GCN_2(torch.nn.Module):
    def __init__(self):
        super(GCN_2, self).__init__()
        dim_h = 60  # Hidden neurons conv layers
        # Convolutional layers
        self.conv1 = GCNConv(16, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.conv4 = GCNConv(dim_h, dim_h)

        # Fully connected layers
        self.lin = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 4)

        # Batch normalization layers
        self.bn1 = BatchNorm1d(dim_h)
        self.bn2 = BatchNorm1d(dim_h)
        self.bn3 = BatchNorm1d(dim_h)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.bn1(h)
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv4(h, edge_index))

        # Graph level readout, pooling function
        hG = global_mean_pool(h, batch)

        # MLP classifier
        hG = F.relu(self.lin(hG))
        hG = self.bn3(hG)
        hG = F.dropout(hG, p=0.3, training=self.training)
        hG = self.lin2(hG)

        return F.log_softmax(hG, dim=1)


class GCN_3(torch.nn.Module):
    def __init__(self):
        super(GCN_3, self).__init__()
        dim_h = 100  # Hidden neurons conv layers
        # Convolutional layers
        self.conv1 = GCNConv(16, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)

        # Fully connected layers
        self.lin = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, dim_h)
        self.lin3 = Linear(dim_h, 4)

        # Batch normalization layers
        self.bn1 = BatchNorm1d(dim_h)
        self.bn2 = BatchNorm1d(dim_h)
        self.bn3 = BatchNorm1d(dim_h)

    def forward(self, x, edge_index, batch):
        # Applying convolutional layer
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=0.05, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)

        # Graph-level readout, pooling function
        hG = global_add_pool(h, batch)

        # MLP classifier
        hG = F.relu(self.lin(hG))
        hG = self.bn3(hG)
        hG = F.relu(self.lin2(hG))
        hG = F.dropout(hG, p=0.35, training=self.training)
        hG = self.lin3(hG)

        return F.log_softmax(hG, dim=1)
