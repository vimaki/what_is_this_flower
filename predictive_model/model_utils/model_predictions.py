"""Making predictions for transmitted data using a neural network.

Functions
---------
run_tta
    Performing a prediction with Test Time Augmentation.
predict
    Make predictions for the transmitted data.
predict_one_sample
    Make predictions for one image.

References
----------
load_dataset.py
    A module that overrides the Dataset class called MyDataset that
    stores images and their labels.
"""

import numpy as np
import torch
import ttach as tta

from .load_dataset import MyDataset


def run_tta(model, inputs: torch.Tensor) -> torch.Tensor:
    """Performing a prediction with Test Time Augmentation.

    Prediction is made for six copies of the transmitted image,
    transformed in different ways and averaged predictions.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    inputs : torch.Tensor
        A three-dimensional matrix that is a digital representation
        of an image to be classified.

    Returns
    -------
    torch.Tensor
        A torch tensor of floats that match the model's confidence
        in assigning each of the classes to the image.
    """

    logits_tta = np.ones((1, list(model.modules())[-1].out_features))
    for transformer in tta.aliases.d4_transform():
        image_tta = transformer.augment_image(inputs)
        outputs_tta = model(image_tta).cpu()
        logits_tta = np.concatenate((logits_tta, outputs_tta), axis=0)
    outputs = np.mean(logits_tta[1:, :], axis=0)
    outputs = torch.from_numpy(outputs)
    return outputs


def predict(model, data: MyDataset, on_gpu: bool = True,
            do_tta: bool = True) -> np.ndarray:
    """Make predictions for the transmitted data.

    The distribution of model confidence over all classes for the
    transmitted images is calculated. It is also possible to make
    predictions using Test Time Augmentation, i.e. prediction is made
    for six copies of the transmitted image, transformed in different
    ways and averaged predictions.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    data : MyDataset
        An object that contains images to be classified.
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).
    do_tta : bool, optional
        Flag that determines whether Test Time Augmentation are required
        (default is True).

    Returns
    -------
    np.ndarray
        A numpy array containing an arrays of floats that match the
        model's confidence in assigning each of the classes to the image.
    """

    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data = [sample[0].unsqueeze(0) for sample in data]
    test_predictions = []
    with torch.no_grad():

        for inputs in data:
            inputs = inputs.to(device)
            model.eval()

            if do_tta:
                outputs = run_tta(model, inputs)
            else:
                outputs = model(inputs).cpu()

            probs = torch.nn.functional.softmax(outputs, dim=0).data.numpy()
            test_predictions.append(probs)

    probs_all = np.array(test_predictions)
    return probs_all


def predict_one_sample(model, inputs: torch.Tensor, on_gpu: bool = True,
                       do_tta: bool = True) -> np.ndarray:
    """Make predictions for one image.

    The distribution of model confidence over all classes for one
    transmitted image is calculated. It is also possible to make
    predictions using Test Time Augmentation, i.e. prediction is made
    for six copies of the transmitted image, transformed in different
    ways and averaged predictions.

    Parameters
    ----------
    model : torch.nn.Module
        A model that is a specific architecture of an artificial neural
        network that runs in the Pytorch framework.
    inputs : torch.Tensor
        A three-dimensional matrix that is a digital representation
        of an image to be classified.
    on_gpu : bool, optional
        Flag that determines whether calculations will be performed
        on a GPU or on a CPU (default is True, i.e. on a GPU).
    do_tta : bool, optional
        Flag that determines whether Test Time Augmentation are required
        (default is True).

    Returns
    -------
    np.ndarray
        A numpy array of floats that match the model's confidence
        in assigning each of the classes to the image.
    """

    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()

        if do_tta:
            outputs = run_tta(model, inputs)
        else:
            outputs = model(inputs).cpu()

        probs = torch.nn.functional.softmax(outputs, dim=-1).numpy()
    return probs
