import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
from typing import List, Tuple

from .load_dataset import MyDataset


def fit_epoch(model, train_loader: DataLoader, loss_func, optimizer,
              on_gpu: bool = True) -> Tuple[float, float]:
    """Passing an epoch in train mode"""
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


def eval_epoch(model, val_loader: DataLoader, loss_func, epoch: int,
               best_acc: float, model_name: str, on_gpu: bool = True)\
        -> Tuple[float, float]:
    """Passing an epoch in evaluate mode"""
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
    if val_acc > best_acc:
        state = {
            'net': model.state_dict(),
            'acc': val_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./gdrive/My Drive/dog_breed/checkpoint'):
            os.mkdir('./gdrive/My Drive/dog_breed/checkpoint')
        torch.save(state,
                   f'./gdrive/My Drive/dog_breed/checkpoint/ckpt_{model_name}.pth')

    return val_loss, val_acc


def train(train_files: MyDataset, val_files: MyDataset, model, loss_func,
          optimizer, scheduler, epochs: int, batch_size: int, model_name: str) \
        -> List[Tuple[float, float, float, float]]:
    """Neural network training"""
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

            train_loss, train_acc = fit_epoch(model, train_loader,
                                              loss_func, optimizer)
            print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, loss_func,
                                           epoch, best_acc, model_name)
            history.append((train_loss, train_acc, val_loss, val_acc))

            best_acc = max(best_acc, val_acc)
            best_epoch = epoch if best_acc == val_acc else best_epoch

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                           v_loss=val_loss, t_acc=train_acc,
                                           v_acc=val_acc))

    return history
