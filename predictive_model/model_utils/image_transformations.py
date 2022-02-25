"""Image transformations for input to a machine learning model.

Functions
---------
load_sample
    Loading images from a file.
crop_image
    Crop images to a square shape.
image_to_tensor
    Transformation of the image to a Pytorch tensor.
"""

from pathlib import Path

from PIL import Image
from torch import Tensor
from torchvision import transforms

__all__ = ['load_sample',
           'crop_image',
           'image_to_tensor']


def load_sample(file: Path) -> Image:
    """Loading images from a file."""
    image = Image.open(file).convert('RGB')
    image.load()
    return image


def crop_image(image: Image) -> Image:
    """Crop images to a square shape."""
    crop_size = int(min(image.size) * 0.95)
    width, height = image.size

    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    image = image.crop((left, top, right, bottom))
    return image


def image_to_tensor(image: Image, mode: str = 'test',
                    rescale_size: int = 224) -> Tensor:
    """Transformation of the image to a Pytorch tensor.

    The image is converted to a tensor. Also, resizing and normalization
    are performed, corresponding to the images on which the neural
    network was pretrained. For image for training and validation,
    additional augmentation is performed.

    Parameters
    ----------
    image : Image
        The sample image is in Pillow Image format.
    mode: str, optional
        One of the three uses of the image ('train', 'val', 'test'),
        i.e. image for training, validation, or testing a machine
        learning model, respectively.
    rescale_size : int, optional
        The size to which the width and height of the image should
        be reduced. Should correspond to the size of the images on
        which the neural network was pretrained (default is 224).

    Returns
    -------
    Tensor
        A three-dimensional Pytorch Tensor that is a digital
        representation of the image.
    """

    if mode != 'test':
        transform = transforms.Compose([
            transforms.Resize((rescale_size, rescale_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0, 0, 0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((rescale_size, rescale_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    return transform(image)
