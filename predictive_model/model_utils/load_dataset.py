"""Loading data into the format required for the neural network.

Classes
-------
MyDataset
    Dataset with images loaded from files.
ClassDistribution
    A helper class for storing the distribution of images into classes.

Functions
---------
upload_dataset
    Loads local files into datasets with images.
create_dct_files_paths
    Mapping to each class a list of paths to files with images.
oversampling
    Extending classes with a small number of images.
"""

from __future__ import annotations
import json
import pickle
from collections import Counter, defaultdict
from itertools import cycle, islice
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

from . import image_transformations


class MyDataset(Dataset):
    """Dataset with images loaded from files.

    Extension of the standard Dataset class from Pytorch. Added such
    capabilities for reading images from files, label encoder,
    augmentation and conversion to the required format, viewing the
    distribution of images by classes.

    Methods
    -------
    __init__
        Inits an extended version of the Pytorch Dataset.
    __len__
        The number of images in the dataset.
    __getitem__
        Convert image to the format required for Pytorch.
    class_distribution
        The number of images for each class.
    """

    def __init__(self, files: List[Path], mode: str,
                 rescale_size: int = 224) -> None:
        """Inits an extended version of the Pytorch Dataset.

        Parameters
        ----------
        files : list[Path]
            A list of paths where images are stored.
        mode : str
            One of the three uses of the dataset ('train', 'val', 'test'),
            i.e. data for training, validation and testing, respectively.
        rescale_size : int, optional
            The size to which the width and height of the image should
            be reduced. Should correspond to the size of the images on
            which the neural network was pretrained (default is 224).

        Returns
        -------
        None
        """

        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.rescale_size = rescale_size

        data_modes = ['train', 'val', 'test']
        if self.mode not in data_modes:
            print(f'{self.mode} is not correct; correct modes: {data_modes}')
            raise NameError

        self.len_ = len(self.files)

        self.labels = [path.parent.name for path in self.files]
        self.label_encoder = LabelEncoder()

        if self.mode == 'train':
            self.label_encoder.fit(self.labels)

            labels_mapping = dict(zip(range(len(self.label_encoder.classes_)),
                                      self.label_encoder.classes_))

            with open('label_encoder.json', 'w') as label_encoder_json_dump_file:
                json.dump(labels_mapping, label_encoder_json_dump_file)

            with open('label_encoder.pkl', 'wb') as label_encoder_pickle_dump_file:
                pickle.dump(self.label_encoder, label_encoder_pickle_dump_file)

        else:
            with open('label_encoder.pkl', 'rb') as label_encoder_pickle_dump_file:
                self.label_encoder = pickle.load(label_encoder_pickle_dump_file)

    def __len__(self):
        """The number of images in the dataset."""
        return self.len_

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Convert image to the format required for Pytorch.

        Extracts a tensor-class pair from the path to the image, in the
        formats necessary for transferring to the input of the artificial
        neural network. For this, resizing, augmentation, type conversion
        and normalization are performed.

        Parameters
        ----------
        index : int
            The index of the corresponding path to the image.

        Returns
        -------
        tuple(Tensor, int)
            A two-element tuple containing a three-dimensional Pytorch
            Tensor representing an image and a corresponding class label.
        """

        x = image_transformations.load_sample(self.files[index])
        x = image_transformations.crop_image(x)
        x = image_transformations.image_to_tensor(
            x, mode=self.mode, rescale_size=self.rescale_size)
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        return x, y

    def class_distribution(self) -> ClassDistribution:
        """The number of images for each class.

        Returns
        -------
        ClassDistribution
            A list of two-element tuples containing the name of the
            classes and the number of corresponding images.
        """

        images_by_class = ClassDistribution()
        number_of_images_by_class = Counter(self.labels)
        for image_class in number_of_images_by_class.items():
            images_by_class.append((image_class[0], image_class[1]))
        return images_by_class


class ClassDistribution(list):
    """A helper class for storing the distribution of images into classes.

    A list of two-element tuples containing the name of the classes and
    the number of corresponding images. When new elements are added,
    it is checked against the pattern. Formatting occurs when printing
    information.

    Methods
    -------
    append
        Append elements only suitable in format.
    __str__
        Formatting distribution of images by classes in the output.
    """

    def append(self, pair: Tuple[str, int]):
        """Append elements only suitable in format."""
        if not isinstance(pair, tuple) and len(pair) != 2:
            raise ValueError('You must pass a tuple of two elements')
        super().append(pair)

    def __str__(self):
        """Formatting distribution of images by classes in the output."""
        representation = []
        max_class_name_length = max(map(lambda x: len(x[0]), self))
        for cls in self:
            representation.append(f'{cls[0]:<{max_class_name_length}}{cls[1]:>7}')
        return '\n'.join(representation)


def upload_dataset(data_directory: str, random_state: int = 0,
                   balance: Optional[int] = None, rescale_size: int = 224) \
        -> Tuple[MyDataset, MyDataset, MyDataset]:
    """Loads local files into datasets with images.

    All the necessary images are found along the passed path and the
    corresponding classes are determined. All data is divided into data
    for training, validation and testing and converted into datasets.

    Parameters
    ----------
    data_directory : str
        The path on the disk where the folders with all the images
        are located.
    random_state : int, optional
        The initial value of the random number generator is fixed for
        the reproducibility of the results obtained (default is 0).
    balance : None or int
        If None is transmitted (default is None), then the number of
        images remains unchanged.
        If an integer is passed, then the number of images in each class
        must be at least this number. Thus, small classes are extended
        by repeating some images.
    rescale_size : int, optional
        The size to which the width and height of the image should be
        reduced. Should correspond to the size of the images on which
        the neural network was pretrained (default is 224).

    Returns
    -------
    Tuple(MyDataset, MyDataset, MyDataset)
        A three-element tuple containing datasets of data for training,
        validation and testing.
    """

    dataset_directory = Path(data_directory)
    dataset_files = sorted(list(dataset_directory.rglob('*.jpg')))

    # extraction of a test sample from a dataset
    train_test_labels = [path.parent.name for path in dataset_files]
    train_files, test_files = train_test_split(
        dataset_files, test_size=0.25,
        stratify=train_test_labels, random_state=random_state
    )

    # extraction of a validation sample from a train data
    train_val_labels = [path.parent.name for path in train_files]
    train_files, val_files = train_test_split(
        train_files, test_size=0.25,
        stratify=train_val_labels, random_state=random_state
    )

    if balance:
        train_files = oversampling(train_files, balance)

    # Convert all samples to pytorch' Datasets
    train_dataset = MyDataset(train_files, mode='train', rescale_size=rescale_size)
    val_dataset = MyDataset(val_files, mode='val', rescale_size=rescale_size)
    test_dataset = MyDataset(test_files, mode='test', rescale_size=rescale_size)

    return train_dataset, val_dataset, test_dataset


def create_dct_files_paths(files: List[Path]) -> Dict[str, List[Path]]:
    """Mapping to each class a list of paths to files with images."""
    labels = [path.parent.name for path in files]
    dct_files_paths = defaultdict(list)
    for path_i, label_i in zip(files, labels):
        dct_files_paths[label_i].append(path_i)
    return dct_files_paths


def oversampling(data: List[Path], n_min: int) -> List[Path]:
    """Extending classes with a small number of images.

    Classes containing the number of images less than n_min will be
    expanded by repeating some of the existing images.

    Parameters
    ----------
    data : list(Path)
        A list of paths to files with images.
    n_min : int
        The number of images in each class must be at least this number.

    Returns
    -------
    list(Path)
        Extended list of paths to files with images.
    """

    dct_files_paths = create_dct_files_paths(data)

    for label in dct_files_paths:
        if len(dct_files_paths[label]) < n_min:
            dct_files_paths[label] = list(islice(cycle(dct_files_paths[label]),
                                                 n_min))

    oversampled_data = []
    for label in dct_files_paths:
        oversampled_data.extend(dct_files_paths[label])
    return oversampled_data
