"""Helper functions for the selection of hyperparameters of neural networks.

Functions
---------
find_init_lr
    Iterating over different values of the step of gradient descent.
plot_lr_loss
    Plot the values of the loss function versus the gradient descent step.
plot_tuning_result
    Plot the history of the selection of hyperparameters.
run_tuning
    Running trials for the selection of hyperparameters.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook

from .load_dataset import MyDataset

__all__ = ['find_init_lr',
           'plot_lr_loss',
           'plot_tuning_result',
           'run_tuning']


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


def plot_tuning_result(study: optuna.study.Study) -> None:
    """Plot the history of the selection of hyperparameters."""
    # Visualize the learning curves of the trials
    fig1 = optuna.visualization.plot_intermediate_values(study)
    # Visualize combinations of hyperparameters for all trials with
    # display of the objective value
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    # The projection onto the surface plane of the objective value
    # depending on all pairs of parameters
    fig3 = optuna.visualization.plot_contour(study)
    # Visualize parameter importances
    fig4 = optuna.visualization.plot_param_importances(study)
    # Visualize which hyperparameters are affecting the trial duration
    # with hyperparameter importance
    fig5 = optuna.visualization.plot_param_importances(
        study,
        target=lambda t: t.duration.total_seconds(),
        target_name='duration'
    )

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()


def run_tuning(objective: Callable[[optuna.trial.Trial], float],
               n_trials: int = 100, seed: int = 0) -> None:
    """Running trials for the selection of hyperparameters.

    A function is launched that contains parameters and has a quality
    assessment metric. A kind of Bayesian search by parameters is
    carried out, in the space described in the function itself.
    At the end, statistics for all trials are displayed.

    Parameters
    ----------
    objective : function
        A function that returns a numerical value to evaluate the
        performance of the hyperparameters, and decide where to sample
        in upcoming trials.
    n_trials : int, optional
        The number of trials that will run with different hyperparameter
        values.
    seed : int, optional
        The initial value of the random number generator is fixed for
        the reproducibility of the results obtained (default is 0).

    Returns
    -------
    None
    """

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed),
                                direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    plot_tuning_result(study)
