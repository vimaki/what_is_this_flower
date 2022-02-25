"""Visualizations associated with printing images.

Functions
---------
imshow_tensor
    Display Pytorch Tensor as an image.
show_dataset_examples
    Display sample images from a dataset.
show_images_with_predictions
    Print images with true and predicted classes.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

from math import ceil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.font_manager import FontProperties
from torch import Tensor

from .load_dataset import MyDataset

__all__ = ['imshow_tensor',
           'show_dataset_examples',
           'show_images_with_predictions']


def imshow_tensor(inp: Tensor, title: bool = None, plt_ax=plt) -> None:
    """Display Pytorch Tensor as an image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_dataset_examples(dataset: MyDataset, n_samples: int = 16) -> None:
    """Display sample images from a dataset."""
    n_rows = ceil(n_samples / 4)
    if n_rows == 1:
        n_columns = n_samples
    else:
        n_columns = 4

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(10, 10),
                           sharey=True, sharex=True)
    for fig_x in ax.flatten():
        random_characters = int(np.random.uniform(0, len(dataset)))
        image, label = dataset[random_characters]
        img_label = ' '.join(map(lambda x: x.capitalize(),
                                 dataset.label_encoder.inverse_transform([label])[0]
                                 .split('_')))
        imshow_tensor(image.data.cpu(), title=img_label, plt_ax=fig_x)


def show_images_with_predictions(dataset: MyDataset, predictions: np.ndarray,
                                 n_samples: int = 9) -> None:
    """Print images with true and predicted classes.

    N images randomly selected from the dataset are printed. Above the
    image, its real class is displayed, and the class predicted by the
    model is applied to the image, as well as the model's confidence in
    its prediction, expressed as a percentage.

    Parameters
    ----------
    dataset : MyDataset
        An object that contains images and their labels.
    predictions : np.ndarray
        Model predictions for passed data. A numpy array containing an
        arrays of floats that match the model's confidence in assigning
        each of the classes to the image.
    n_samples : int
        The number of randomly selected images for which classification
        information will be displayed.

    Returns
    -------
    None
    """

    n_rows = ceil(n_samples / 3)
    if n_rows == 1:
        n_columns = n_samples
    else:
        n_columns = 3

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(12, 12),
                           sharey=True, sharex=True)
    for fig_x in ax.flatten():
        random_character = int(np.random.uniform(0, len(dataset)))
        image, label = dataset[random_character]
        img_label = ' '.join(map(lambda x: x.capitalize(),
                                 dataset.label_encoder.inverse_transform([label])[0]
                                 .split('_')))
        ground_truth = 'Actual : {}'.format(img_label)

        imshow_tensor(image.data.cpu(), title=ground_truth, plt_ax=fig_x)

        fig_x.add_patch(patches.Rectangle((0, 53), 86, 20, color='white'))
        font0 = FontProperties()
        font = font0.copy()

        image_prediction = predictions[random_character]
        predicted_proba = np.max(image_prediction) * 100
        class_prediction = np.argmax(image_prediction)

        predicted_label = dataset.label_encoder.classes_[class_prediction]
        predicted_text = '{} : {:.0f}%'.format(predicted_label, predicted_proba)

        fig_x.text(1, 59, predicted_text, horizontalalignment='left',
                   fontproperties=font, verticalalignment='top',
                   fontsize=8, color='black', fontweight='bold')
