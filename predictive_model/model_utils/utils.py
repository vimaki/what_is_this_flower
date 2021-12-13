"""Helper functions for training artificial neural networks.

Functions
---------
find_init_lr
    Iterating over different values of the step of gradient descent.
plot_lr_loss
    Plot the values of the loss function versus the gradient descent step.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from typing import List, Tuple

from .load_dataset import MyDataset


def find_init_lr(model, train_dataset: MyDataset, loss_func, optimizer,
                 init_value: float = 1e-8, final_value: float = 10.0,
                 batch_size: int = 64, on_gpu: bool = True) \
        -> Tuple[List[float], List[float]]:
    """Iterating over different values of the step of gradient descent.

    One epoch is produced on the dataset for training. For each new
    sample, a new gradient descent value is set, depending on the
    initial value, the final value and the size of the dataset. All
    values of the gradient descent step and the corresponding values
    of the loss function are recorded.
    The possibility of early stopping of the algorithm in case of a sharp
    increase in the value of the loss function has been implemented.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    train_dataset: MyDataset
        An object that contains images and their labels for training
        the model.
    loss_func : torch.nn.Module
        A function whose values indicate how well the model predicts.
    optimizer: torch.optim.Optimizer
        An algorithm that alters the operation of gradient descent
        to speed up the convergence to a minimum of the loss function.
    init_value: float, optional
        The initial value of the gradient descent step from which the
        search will start (default is 1e-8).
    final_value: float, optional
        The final value of the gradient descent step up to which the
        search will be performed (default is 10.0).
    batch_size : int, optional
        The size of the portions of images that will be simultaneously
        processed by the model (default if 64 samples).
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).

    Returns
    -------
    tuple(list(float), list(float))
        A two-element tuple containing a list of decimal logarithms
        of all applied gradient descent step values and a list of
        corresponding loss function values.
    """

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
    """Plot the values of the loss function versus the gradient descent step."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(logs_lr, losses)
    ax.set_xlabel('lr, $10^x$')
    ax.set_ylabel('loss')
