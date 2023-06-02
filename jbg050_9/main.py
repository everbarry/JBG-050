import numpy as np
import pandas as pd
from typing import *
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torchsummary import summary
import wandb
# modules
from models import *
from dataset import CrimeDataset, CrimeDatasetSW
from parser import parser
from train_test import train, train_sw



def initEnv(random_seed=69)-> Dict:
    """
    function that initializes and return device, sets random seeed for repeatable results.
    """
    settings = vars(parser())
    if settings['wandb']:
        #wandb.login()
        run = wandb.init(project="JBG050", config=settings)
    settings['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# if cuda gpu available use that
    settings['random_seed'] = random_seed
    # set fixed random seed for repeatability
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f'initializing model using {settings["device"]}, random seed: {random_seed}')
    return settings



def main(settings: Dict):
    """
    main that runs train / eval based on given settings
    """
    dataset  = CrimeDatasetSW(root='../data', sequence_length=settings['sliding_window'])
    dataloader = DataLoader(dataset, batch_size=1)

    model = DeeperAttentionGCNSW(node_features=16, filters1=16, filters2=32, heads1=10, heads2=10)
    #model = AttentionGCNSW(node_features=16, filters=1, heads=10)
    model.to(settings['device'])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'], amsgrad=settings['amsgrad'])

    if settings['model_summary']:
        raise NotImplementedError
        summary(model, (24,13692, 16), device=settings['device'])

    if settings['val']:
        raise NotImplementedError

    # train & test model
    results = train_sw(settings, dataloader, optimizer, model, criterion)
    return results

if __name__ == '__main__':
    main(initEnv())
