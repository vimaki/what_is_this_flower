import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from typing import List, Tuple

from .load_dataset import MyDataset


def find_lr(model, train_dataset: MyDataset, loss_func, optimizer,
            init_value: float = 1e-8, final_value: float = 10.0,
            batch_size: int = 64, on_gpu: bool = True)\
        -> Tuple[List[float], List[float]]:
    """Iterating over different values of the step of gradient descent"""
    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in tqdm_notebook(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_num += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs, losses
            # return log_lrs[10:-5], losses[10:-5]

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss)
        log_lrs.append(np.log10(lr))

        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


def plot_lr_loss(logs_lr: List[float], losses: List[float]) -> None:
    """Plot the changes of loss function versus the steps of gradient descent"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(logs_lr, losses)
    ax.set_xlabel('lr, $10^x$')
    ax.set_ylabel('loss')
