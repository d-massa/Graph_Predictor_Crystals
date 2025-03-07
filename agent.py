from torch_geometric.data import DataLoader
from models import GCN, GNN_SAGE
import tqdm
import torch
import wandb
from data import JarvisDataset, dataset

def agent_fn():
    wandb.init()

    torch.manual_seed(176432)
    train_size = int(wandb.config.TrainS *len(dataset))
    test_size = len(dataset)-train_size

    train_dataset = JarvisDataset(dataset[:train_size])
    
    test_dataset = JarvisDataset(dataset[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.BatchS, shuffle=True) # Shuffle for training
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.BatchS, shuffle=False) # Shuffle for training

    if wandb.config.model == "GCN": 
        model = GCN(hidden_channels = wandb.config.hidden_channels,
                    num_layers=wandb.config.num_layers)
    elif wandb.config.model == "GNN_SAGE":
        model = GNN_SAGE(hidden_channels = wandb.config.hidden_channels,
                         num_layers=wandb.config.num_layers,
                         aggr=wandb.config.get("aggr", "mean"),  # defaults to 'mean'
                         aggr_kwargs=wandb.config.get("aggr_kwargs", None))
    else: 
        ValueError("UNKOWN")

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x,data.edge_index,data.batch)
            loss = criterion(out, data.y)            
            loss.backward()
            optimizer.step()
           # return loss

    def test(loader):
        model.eval()

        for data in loader:
            out = model(data.x,data.edge_index,data.batch)
            loss = criterion(out, data.y)            
            return loss 

    for epoch in tqdm.tqdm(range(1,wandb.config.epochs+1)):
        train()

        train_loss = test(train_loader)
        test_loss = test(test_loader)
        
        train_rmse_loss =  torch.sqrt(train_loss.detach().mean())
        test_rmse_loss = torch.sqrt(test_loss.detach().mean())
        scale_tensor_train= torch.tensor(train_dataset.scale(), dtype=torch.float32)
        scale_tensor_test= torch.tensor(test_dataset.scale(), dtype=torch.float32)
        #print(f"\n SCALING FACTOR: {scale_tensor_train},{scale_tensor_test}\n")

        #mean_tensor = torch.tensor(train_dataset.mean(), dtype=torch.float32)
        train_mse_loss_original = torch.exp(train_loss) * torch.exp(scale_tensor_train ** 2) - 1.0
        train_rmse_loss_original = torch.sqrt(train_mse_loss_original.mean())
        test_mse_loss_original = torch.exp(test_loss) * torch.exp(scale_tensor_test ** 2) - 1.0
        test_rmse_loss_original = torch.sqrt(test_mse_loss_original.mean())

        wandb.log({f"gcn_jarvis-ADV-{wandb.config.model}"+"/train_RMSE": train_rmse_loss.item()})
        wandb.log({f"gcn_jarvis-ADV-{wandb.config.model}"+"/test_RMSE": test_rmse_loss.item()})
        wandb.log({f"gcn_jarvis-ADV-{wandb.config.model}"+"/train_RMSE_orig": train_rmse_loss_original.item()})
        wandb.log({f"gcn_jarvis-ADV-{wandb.config.model}"+"/test_RMSE_orig": test_rmse_loss_original.item()})

    wandb.finish()