"""Checking the quality of the model.

Functions
---------
plot_epoch_loss
    Plot the error changes versus the number of epochs.
model_f1
    Calculate F-score based on the transmitted predictions.
plot_confusion_matrix
    Prints a confusion matrix of images classification.
show_confusion_matrix_func
    Display the distribution of accuracy by class as a confusion matrix.
show_accuracy_for_each_class
    Display accuracy for each class.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from typing import List, Tuple

from .load_dataset import MyDataset


def plot_epoch_loss(loss: Tuple[float], val_loss: Tuple[float]) -> None:
    """Plot the error changes versus the number of epochs."""
    plt.figure(figsize=(15, 9))
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def model_f1(data: MyDataset, predictions: np.ndarray) -> float:
    """Calculate F-score based on the transmitted predictions."""
    y_pred = np.argmax(predictions, axis=-1)
    y_ground_truth = [example[1] for example in data]
    return f1_score(y_ground_truth, y_pred, average='micro')


def plot_confusion_matrix(conf_mat: np.ndarray, classes: List[str],
                          normalize: bool = False, title: str = 'Confusion matrix',
                          cmap='Blues') -> None:
    """Prints a confusion matrix of images classification.

    A table is printed with the same number of rows and columns equal
    to the number of classes. In this table i-th row and j-th column
    entry indicates the number of samples with true label being i-th
    class and predicted label being j-th class. The numbers in the table
    cells can be either an absolute number of samples or normalized.

    Parameters
    ----------
    conf_mat : np.ndarray
        A numpy array of shape (n_classes, n_classes) corresponding
        to the confusion matrix.
    classes : list(str)
        A list containing the names of all classes.
    normalize : bool, optional
        Flag for displaying normalized data (default is False). This
        means that all values for each class will be scaled from 0 to 1.
    title : str, optional
        The title of the figure that explains its contents (default is
        'Confusion matrix').
    cmap : str, optional
        The name of the colormap that will be applied when printing
        the figure. The name should be taken from the list of colormaps
        in the matplotlib library (default is 'Blues').

    Returns
    -------
    None
    """

    conf_mat = conf_mat.T
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(16, 11))
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]),
                                  range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()


def show_confusion_matrix_func(data: MyDataset, predictions: np.ndarray,
                               plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Display the distribution of accuracy by class as a confusion matrix.

    A confusion matrix is a two-dimensional array of dimensions
    (n_classes, n_columns) whose i-th row and j-th column entry indicates
    the number of samples with true label being i-th class and predicted
    label being j-th class

    Parameters
    ----------
    data : MyDataset
        The data for which the precision distribution will be calculated.
    predictions : np.ndarray
        Model predictions for passed data. A numpy array containing an
        arrays of floats that match the model's confidence in assigning
        each of the classes to the image.
    plot: bool, optional
        A flag indicating whether to print the confusion matrix.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        A tuple containing 2 numpy arrays. The first stores the true
        classes of the transferred images, and the second stores the
        classes predicted by the model.
    """

    class_names = sorted(set(data.labels))

    y_ground_truth = np.array([sample[1] for sample in data])
    y_pred = np.argmax(predictions, axis=-1)

    conf_mat = confusion_matrix(y_ground_truth, y_pred,
                                labels=np.arange(len(class_names)))
    if plot:
        plot_confusion_matrix(conf_mat, class_names, normalize=True)

    return y_ground_truth, y_pred


def show_accuracy_for_each_class(data: MyDataset, predictions: np.ndarray) -> None:
    """Display accuracy for each class.

    Parameters
    ----------
    data : MyDataset
        The data for which the precision distribution will be calculated.
    predictions : np.ndarray
        Model predictions for passed data. A numpy array containing an
        arrays of floats that match the model's confidence in assigning
        each of the classes to the image.

    Returns
    -------
    None
    """

    class_names = sorted(set(data.labels))
    max_class_name_length = max(map(lambda x: len(x), class_names))

    y_ground_truth = np.array([example[1] for example in data])
    y_pred = np.argmax(predictions, axis=-1)

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    correct_answer = (y_pred.squeeze() == y_ground_truth)
    for i in range(len(y_pred)):
        label = y_pred[i].item()
        class_correct[label] += correct_answer[i]
        class_total[label] += 1

    for i in range(len(class_names)):
        percentage = ((100 * class_correct[i] / class_total[i])
                      if class_total[i] != 0 else -1)
        print(f'Accuracy of {class_names[i]:<{max_class_name_length}} {percentage:>5.1f}%')
