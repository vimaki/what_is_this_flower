"""Performing flower classification on the transferred image.

Functions
---------
transform_image
    Image transformation to transfer it to the classification model.
get_inference
    Determination of the type of flower shown in the image.

References
----------
load_dataset.py
    A module that contains functions for reading and converting images
    that are inside the class MyDataset.
model_weights.pth
    A file containing the weights of the trained neural network model.
label_encoder.json
    A file containing mappings predicted labels to flower names.
flower_types.json
    A file containing mappings English names to Russian names of flowers.
"""

import json
import sys
from typing import Tuple

from torch import device, load, Tensor
from torch.nn import Linear
from torchvision import models

# Loading functions from an external package
sys.path.insert(1, '../predictive_model/model_utils')
import image_transformations

PATH_TO_MODEL = '../predictive_model/model_weights.pth'
LABEL_ENCODER = '../predictive_model/label_encoder.json'
FLOWER_DICTIONARY = '../scraping_dataset/flower_types.json'

# Loading mappings predicted labels to flower names
with open(LABEL_ENCODER) as f:
    label_encoder = json.load(f)

# Loading mappings English names to Russian names of flowers
with open(FLOWER_DICTIONARY) as f:
    flower_dict = json.load(f)

# Loading the classification model
model = models.efficientnet_b5()
model.classifier[1] = Linear(model.classifier[1].in_features, len(flower_dict))
model.load_state_dict(load(PATH_TO_MODEL, map_location=device('cpu')))
model.eval()


def transform_image(image_path: str) -> Tensor:
    """Image transformation to transfer it to the classification model.

    Reads images from a path in the file system. Then the image is
    transformed to be processed by the classification model. Resizing,
    converting to a tensor and normalizing are performed to do this.

    Parameters
    ----------
    image_path : str
        The path where the image is stored.

    Returns
    -------
    Tensor
        A three-dimensional Pytorch Tensor representing an image.
    """

    image = image_transformations.load_sample(image_path)
    image = image_transformations.crop_image(image)
    image = image_transformations.image_to_tensor(
        image, mode='test', rescale_size=224).unsqueeze(0)
    return image


def get_inference(image_path: str) -> Tuple[str, str]:
    """Determination of the type of flower shown in the image.

    Parameters
    ----------
    image_path : str
        The path where the image is stored.

    Returns
    -------
    tuple(str, str)
        A two-element tuple containing the predicted name of the flower
        type in English and Russian.
    """

    # Making a prediction by the model
    image = transform_image(image_path)
    outputs = model.forward(image)
    _, y_hat = outputs.max(1)
    predicted_label = str(y_hat.item())

    # Getting the names of flowers
    flower_name_eng = label_encoder[predicted_label]
    flower_name_rus = flower_dict[flower_name_eng]
    return flower_name_eng, flower_name_rus
