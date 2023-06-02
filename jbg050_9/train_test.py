from typing import *
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm
import wandb


def train(settings, dataloader, optimizer, model, loss_fn)-> Dict:
    y_hat, y_true, losses, r2_scores = [], [], [], []
    for epoch in tqdm(range(settings['epochs']), desc='training...'):
        epoch_losses, epoch_r2_scores = [], []
        for data in dataloader:
            # Extract the data
            x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr.float(), data.y
            x, edge_index, edge_attr, y = x.to(settings['device']), edge_index.to(settings['device']), edge_attr.to(settings['device']), y.to(settings['device'])
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            H = model(x, edge_index)
            #H = model(x, edge_index, edge_attr)            # H, c
            # Compute the loss
            loss = loss_fn(H, y.unsqueeze(1).to(settings['device']))
            epoch_losses.append(loss.item())
            # Compute metrics
            r2_score_value = r2_score(y.cpu().numpy(), H.detach().cpu().numpy())
            epoch_r2_scores.append(r2_score_value)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            if epoch == settings['epochs'] - 1: # save last predictions
                y_hat.append(H.detach().cpu().numpy())
                y_true.append(y.unsqueeze(1).detach().cpu().numpy())
        # log stuff
        losses.append(np.mean(epoch_losses))
        r2_scores.append(np.mean(epoch_r2_scores))
        if settings['wandb']:
            wandb.log({"r_squared": r2_scores[-1], "loss": losses[-1]})
        if epoch % 2 == 0:
            tqdm.write(f"{'-'*20}\nEpoch : {epoch}\nLoss (MSE) = {losses[-1]:.4f}\nR2 Score = {r2_scores[-1]:.4f}")

    return {'y': pd.DataFrame(np.concatenate(y_true, axis=1).T), 'y_hat': pd.DataFrame(np.concatenate(y_hat, axis=1).T), 'r2s': r2_scores, 'losses': losses}


def train_sw(settings, dataloader, optimizer, model, loss_fn)-> Dict:
    y_hat, y_true, losses, r2_scores = [], [], [], []
    for epoch in tqdm(range(settings['epochs']), desc='training...'):
        epoch_losses, epoch_r2_scores = [], []
        for data in dataloader:
            # Extract the data
            x_sequence = data.x.to(settings['device'])
            edge_index_sequence = [edge_index.to(settings['device']) for edge_index in data.edge_index]
            y = data.y.to(settings['device'])

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            H = model(x_sequence, edge_index_sequence)

            # Compute the loss
            loss = loss_fn(H, y)
            epoch_losses.append(loss.item())

            # Compute metrics
            r2_score_value = r2_score(y.cpu().numpy(), H.detach().cpu().numpy())
            epoch_r2_scores.append(r2_score_value)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            if epoch == settings['epochs'] - 1: # save last predictions
                y_hat.append(H.detach().cpu().numpy())
                y_true.append(y.cpu().numpy())

        # log stuff
        losses.append(np.mean(epoch_losses))
        r2_scores.append(np.mean(epoch_r2_scores))

        if settings['wandb']:
            wandb.log({"r_squared": r2_scores[-1], "loss": losses[-1]})

        if epoch % 2 == 0:
            tqdm.write(f"{'-'*20}\nEpoch : {epoch}\nLoss (MSE) = {losses[-1]:.4f}\nR2 Score = {r2_scores[-1]:.4f}")

    return {'y': pd.DataFrame(np.concatenate(y_true).reshape(-1, 1)),
            'y_hat': pd.DataFrame(np.concatenate(y_hat).reshape(-1, 1)),
            'r2s': r2_scores,
            'losses': losses}
