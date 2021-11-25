import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import ceil
from matplotlib.font_manager import FontProperties
from torch import Tensor

from .load_dataset import MyDataset


def imshow_tensor(inp: Tensor, title: bool = None, plt_ax=plt) -> None:
    """Analogue of the Imshow function for tensors"""
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
    """Display sample images from a dataset"""
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
                                 n_samples=9) -> None:
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

        fig_x.add_patch(patches.Rectangle((0, 53), 86, 35, color='white'))
        font0 = FontProperties()
        font = font0.copy()
        font.set_family('fantasy')

        # prob_pred = predict_one_sample(model_ensemble, im_val.unsqueeze(0))
        image_prediction = predictions[random_character]
        predicted_proba = np.max(image_prediction) * 100
        class_prediction = np.argmax(image_prediction)

        predicted_label = dataset.label_encoder.classes_[class_prediction]
        # predicted_label = (predicted_label[:len(predicted_label) // 2] +
        #                    '\n' + predicted_label[len(predicted_label) // 2:])
        predicted_text = '{} : {:.0f}%'.format(predicted_label, predicted_proba)

        fig_x.text(1, 59, predicted_text, horizontalalignment='left',
                   fontproperties=font, verticalalignment='top',
                   fontsize=8, color='black', fontweight='bold')
