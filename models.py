from torch_geometric.nn import GCNConv,global_mean_pool,SAGEConv
import torch.nn.functional as F
from torch.nn import Linear
import copy
import torch
from data import dataset

class GCN(torch.nn.Module):
    def __init__(self,hidden_channels,num_layers):
        super(GCN,self).__init__()
        torch.manual_seed(176432)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features,hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))
        self.lin = Linear(hidden_channels,1)
    
    def forward(self,x,edge_index,batch):
        for conv in self.convs:
            x = conv(x,edge_index)
            x = F.relu(x)
        x = global_mean_pool(x,batch)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.lin(x)
        return x.squeeze(-1)
    
class GNN_SAGE(torch.nn.Module):
    def __init__(self,hidden_channels,num_layers,aggr='mean',aggr_kwargs=None):
        super(GNN_SAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(dataset.num_features,hidden_channels,aggr=aggr,aggr_kwargs=aggr_kwargs))
        for _ in range (num_layers -1):
            self.convs.append(SAGEConv(hidden_channels,hidden_channels,aggr=copy.deepcopy(aggr),aggr_kwargs=aggr_kwargs))
        self.lin = Linear(hidden_channels,1)

    def forward(self,x,edge_index,batch):
        for conv in self.convs:
            x = conv(x,edge_index)
            x = F.relu(x)
        x = global_mean_pool(x,batch)
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.lin(x)
        return x.squeeze(-1)