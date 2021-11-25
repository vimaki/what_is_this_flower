import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from typing import List, Tuple

from .load_dataset import MyDataset


def plot_epoch_loss(loss: Tuple[float], val_loss: Tuple[float]) -> None:
    """Plot the error changes versus the number of epochs"""
    plt.figure(figsize=(15, 9))
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def model_f1(data: MyDataset, predictions: np.ndarray) -> float:
    """Calculate F-score based on the transmitted predictions"""
    y_pred = np.argmax(predictions, axis=-1)
    y_ground_truth = [example[1] for example in data]
    # preds_class = [i for i in y_pred]
    return f1_score(y_ground_truth, y_pred, average='micro')  # average='weighted'


def plot_confusion_matrix(conf_mat: np.ndarray, classes: List[str],
                          normalize: bool = False, title: str = 'Confusion matrix',
                          cmap=plt.cm.Blues) -> None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    conf_mat = conf_mat.T
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(conf_mat)
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
                               plot: bool = True) -> Tuple[List[int], np.ndarray]:
    """Display the distribution of accuracy by class as a confusion matrix"""
    class_names = sorted(set(data.labels))

    y_pred = np.argmax(predictions, axis=-1)
    y_ground_truth = [sample[1] for sample in data]

    conf_mat = confusion_matrix(y_ground_truth, y_pred,
                                np.arange(len(class_names)))
    if plot:
        plot_confusion_matrix(conf_mat, class_names, normalize=True)

    return y_ground_truth, y_pred


def show_accuracy_for_each_class(data: MyDataset, predictions: np.ndarray) -> None:
    """Display accuracy for each class"""
    class_names = sorted(set(data.labels))
    max_class_name_length = max(map(lambda x: len(x), class_names))

    y_pred = np.argmax(predictions, axis=-1)
    y_ground_truth = [example[1] for example in data]

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    correct_answer = (y_pred == y_ground_truth).squeeze()
    for i in range(len(y_pred)):
        label = y_pred[i]
        class_correct[label] += correct_answer[i].item()
        class_total[label] += 1

    for i in range(len(class_names)):
        percentage = ((100 * class_correct[i] / class_total[i])
                      if class_total[i] != 0 else -1)
        print(f'Accuracy of {class_names[i]:<{max_class_name_length}} {percentage:>3}%')
