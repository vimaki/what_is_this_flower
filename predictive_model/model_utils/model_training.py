"""Neural network training.

Changing the parameters of an artificial neural network based on the
data transmitted to it.

Functions
---------
fit_epoch
    Passing an epoch in train mode.
eval_epoch
    Passing an epoch in evaluate mode.
train
    Neural network training.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

from .load_dataset import MyDataset

__all__ = ['fit_epoch',
           'eval_epoch',
           'train']


def fit_epoch(model, train_loader: DataLoader, loss_func, optimizer,
              on_gpu: bool = True) -> Tuple[float, float]:
    """Passing an epoch in train mode.

    This function transmits a batch of data for training to the input
    of the artificial neural network, receives predictions and calculates
    the value of the loss function. Then, based on the derivatives
    of the loss function over all network weights, a gradient descent
    step is made. The described steps are repeated for all batches of data.
    Loss function and accuracy values for the entire epoch are also stored.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    train_loader : DataLoader
        An object that contains images and their labels for training
        the model and outputs data in certain batches.
    loss_func : torch.nn.Module
        A function whose values indicate how well the model predicts.
    optimizer: torch.optim.Optimizer
        An algorithm that alters the operation of gradient descent
        to speed up the convergence to a minimum of the loss function.
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).

    Returns
    -------
    tuple(float, float)
        A tuple of length 2, which contains the value of the loss
        function and the accuracy calculated for the entire epoch
        and on the training data.
    """

    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in tqdm_notebook(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader: DataLoader, loss_func, on_gpu: bool = True,
               checkpoint: bool = True, epoch: Optional[int] = None,
               best_acc: Optional[float] = None,
               model_name: Optional[str] = None) -> Tuple[float, float]:
    """Passing an epoch in evaluate mode.

    This function transmits a batch of data for validating to the input
    of the artificial neural network, receives predictions and calculates
    the value of the loss function. Thus, we evaluate the quality
    of the model on data that were not involved in setting the weights
    of the network, so we get a more plausible estimate. The described
    steps are repeated for all batches of data. Loss function and
    accuracy values for the entire epoch are also stored.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    val_loader : DataLoader
        An object that contains images and their labels for validating
        the model and outputs data in certain batches.
    loss_func : torch.nn.Module
        A function whose values indicate how well the model predicts.
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).
    checkpoint : bool, optional
        A flag that determines whether to save model checkpoints at
        epochs where the best quality was achieved on validation
        (default is True).
    epoch: None or int
        The current epoch number, i.e. iteration number of the algorithm
        passing through all data.
    best_acc : None or float
        Best accuracy obtained in previous epochs. It is necessary to
        save the checkpoint of the model if the accuracy at this epoch
        is better than the passed value.
    model_name : None or str
        The name of the model type to form the file name of the saved
        checkpoint.

    Returns
    -------
    tuple(float, float)
        A tuple of length 2, which contains the value of the loss
        function and the accuracy calculated for the entire epoch
        and on the validating data.
    """

    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in tqdm_notebook(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            predictions = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size

    # Create checkpoints of models with the best score
    if checkpoint:
        if val_acc > best_acc:
            state = {
                'net': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('./gdrive/My Drive/flowers/checkpoint'):
                os.mkdir('./gdrive/My Drive/flowers/checkpoint')
            torch.save(state,
                       f'./gdrive/My Drive/flowers/checkpoint/ckpt_{model_name}.pth')

    return val_loss, val_acc


def train(train_files: MyDataset, val_files: MyDataset, model, loss_func,
          optimizer, scheduler, epochs: int, batch_size: int, model_name: str,
          on_gpu: bool = True, checkpoint: bool = True) \
        -> List[Tuple[float, float, float, float]]:
    """Neural network training.

    This function starts a two-stage loop consisting of training
    the artificial neural network by changing the coefficients
    of the model based on the transmitted data and then checking
    the quality of the model. At each iteration, network performance
    indicators are saved.

    Parameters
    ----------
    train_files: MyDataset
        An object that contains images and their labels for training
        the model.
    val_files: MyDataset
        An object that contains images and their labels for validating
        the model.
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    loss_func : torch.nn.Module
        A function whose values indicate how well the model predicts.
    optimizer: torch.optim.Optimizer
        An algorithm that alters the operation of gradient descent
        to speed up the convergence to a minimum of the loss function.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        An algorithm to adjust the learning rate based on the number
        of epochs. This makes it possible to more accurately search
        for the minimum of the loss function.
    epochs: int
        The number of iterations that the algorithm goes through all
        the data.
    batch_size : int
        The size of the portions of images that will be simultaneously
        processed by the model.
    model_name : str
        The name of the model type to form the file name of the saved
        checkpoint.
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).
    checkpoint : bool, optional
        A flag that determines whether to save model checkpoints at
        epochs where the best quality was achieved on validation
        (default is True).

    Returns
    -------
    list(tuple(float, float, float, float))
        A list of tuples of length 4, which contains the value of the
        loss function and the accuracy calculated for each epoch on the
        training and validating data.
    """

    train_loader = DataLoader(train_files, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_files, batch_size=batch_size, shuffle=False)

    best_acc = 0
    best_epoch = -1

    history = []
    log_template = '\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}'

    with tqdm(desc='epoch', total=epochs) as pbar_outer:

        for epoch in tqdm_notebook(range(epochs)):
            scheduler.step()

            train_loss, train_acc = fit_epoch(model, train_loader, loss_func,
                                              optimizer, on_gpu=on_gpu)
            print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, loss_func,
                                           on_gpu, checkpoint, epoch,
                                           best_acc, model_name)
            history.append((train_loss, train_acc, val_loss, val_acc))

            best_acc = max(best_acc, val_acc)
            best_epoch = epoch if best_acc == val_acc else best_epoch

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                           v_loss=val_loss, t_acc=train_acc,
                                           v_acc=val_acc))

    return history
