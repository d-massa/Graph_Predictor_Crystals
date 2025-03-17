from agent import agent_fn

enable_wandb = True
if enable_wandb:
    import wandb
wandb.login()
assert enable_wandb, "W&B not enabled. Please, enable W&B and restart the notebook"


project_name =f"jarvis-node-regression-ADV-GNN_SAGE-II"

sweep_config_GNN_SAGE = {
    "name": "gcn_jarvis-ADV-GNN_SAGE-II-v1",
    "method": "bayes",
    "metric": {
        "name": "gcn_jarvis-ADV-GNN_SAGE/test_RMSE",
        "goal": "minimize",
    },
    "parameters": {
        "model":{ "values":["GNN_SAGE"]},
        "hidden_channels": {
            "values": [32,64]
        },
        "num_layers":{
            "values":[3,4,5]
        },
        "BatchS":{
            "values":[128,256]
        },
        "TrainS":{
            "values":[0.85]
        },
        "weight_decay": {
            "distribution": "normal",
            "mu": 1e-5,
            "sigma": 5e-6
        },
        "lr": {
            "min": 1e-5,
            "max": 1e-3
        },
        "aggr": {
            "values": ["min"]
            },
        "epochs":{
            "values":[80]
        }
}
}

project_name =f"jarvis-node-regression-ADV-GATv2"

sweep_config_GATv2 = {
    "name": "gcn_jarvis-ADV-GATv2-v1",
    "method": "bayes",
    "metric": {
        "name": "gcn_jarvis-ADV-GATv2/test_RMSE",
        "goal": "minimize",
    },
    "parameters": {
        "model":{ "values":["GATv2"]},
        "hidden_channels": {
            "values": [32,64]
        },
        "num_layers":{
            "values":[2] # constraint on heads
        },
        "BatchS":{
            "values":[128,256]
        },
        "TrainS":{
            "values":[0.85]
        },
        "weight_decay": {
            "distribution": "normal",
            "mu": 1e-5,
            "sigma": 5e-6
        },
        "lr": {
            "min": 1e-5,
            "max": 1e-3
        },
        "aggr": {
            "values": ["min"]
            },
        "epochs":{
            "values":[300]
        },
        "heads":{
            "values":[[8,1]]
        }
}
}

# Register the Sweep with W&B
sweep_id = wandb.sweep(sweep_config_GATv2, project=project_name)

# Run the Sweeps agent
wandb.agent(sweep_id, project=project_name, function=agent_fn, count=1)