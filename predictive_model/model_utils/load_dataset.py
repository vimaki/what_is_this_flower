from __future__ import annotations
import pickle
from collections import Counter, defaultdict
from itertools import cycle, islice
from pathlib import Path
from PIL import Image, PyAccess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Optional, Tuple, Union


class MyDataset(Dataset):
    """Dataset with images loaded from files"""

    def __init__(self, files: List[Path], mode: str,
                 rescale_size: int = 224) -> None:
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.rescale_size = rescale_size

        data_modes = ['train', 'val', 'test']
        if self.mode not in data_modes:
            print(f'{self.mode} is not correct; correct modes: {data_modes}')
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':  # мной было исправлено на True
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as label_encoder_dump_file:
                pickle.dump(self.label_encoder, label_encoder_dump_file)

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file: Path) -> PyAccess:
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Convert image to the format required for Pytorch"""
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        if self.mode != 'test':
            transform = transforms.Compose([
                transforms.Resize((self.rescale_size, self.rescale_size)),
                # transforms.ToPILImage(),
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
                transforms.Resize((self.rescale_size, self.rescale_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        x = self.load_sample(self.files[index])
        # x = self._prepare_sample(x)
        # x = np.array(x / 255, dtype='float32')
        x = transform(x)
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        return x, y
#         if self.mode == 'test':
#             return x
#         else:
#             label = self.labels[index]
#             label_id = self.label_encoder.transform([label])
#             y = label_id.item()
#             return x, y

    def class_distribution(self) -> ClassDistribution:
        images_by_class = ClassDistribution()
        number_of_images_by_class = Counter(self.labels)
        for image_class in number_of_images_by_class.items():
            images_by_class.append((image_class[0], image_class[1]))
        return images_by_class


class ClassDistribution(list):

    def append(self, pair: Union[Tuple[str, int], List[str, int]]):
        if not isinstance(pair, (tuple, list)) and len(pair) != 2:
            raise ValueError('You must pass a tuple or list of two elements')
        super().append(pair)

    def __str__(self):
        representation = []
        max_class_name_length = max(map(lambda x: len(x[0]), self))
        for cls in self:
            representation.append(f'{cls[0]:<{max_class_name_length}}{cls[1]:>7}')
        return '\n'.join(representation)


def upload_dataset(data_directory: str, balance: Optional[int] = None) \
        -> Tuple[MyDataset, MyDataset, MyDataset]:
    dataset_directory = Path(data_directory)
    dataset_files = sorted(list(dataset_directory.rglob('*.jpg')))

    # extraction of a test sample from a dataset
    train_test_labels = [path.parent.name for path in dataset_files]
    train_files, test_files = train_test_split(
        dataset_files, test_size=0.25, stratify=train_test_labels, random_state=0
    )

    # extraction of a validation sample from a train data
    train_val_labels = [path.parent.name for path in train_files]  # ??? нужно ли ???
    train_files, val_files = train_test_split(
        train_files, test_size=0.25, stratify=train_val_labels, random_state=0
    )

    if balance:
        train_files = oversampling(train_files, balance)

    # Convert all samples to pytorch' Datasets
    train_dataset = MyDataset(train_files, mode='train')
    val_dataset = MyDataset(val_files, mode='val')
    test_dataset = MyDataset(test_files, mode='test')

    return train_dataset, val_dataset, test_dataset


def create_dct_files_paths(files: List[Path]) -> Dict[str, List[Path]]:
    labels = [path.parent.name for path in files]
    dct_files_paths = defaultdict(list)
    for path_i, label_i in zip(files, labels):
        dct_files_paths[label_i].append(path_i)
    return dct_files_paths


def oversampling(data: List[Path], n_min: int) -> List[Path]:
    dct_files_paths = create_dct_files_paths(data)

    for label in dct_files_paths:
        if len(dct_files_paths[label]) < n_min:
            dct_files_paths[label] = list(islice(cycle(dct_files_paths[label]),
                                                 n_min))

    oversampled_data = []
    for label in dct_files_paths:
        oversampled_data.extend(dct_files_paths[label])
    return oversampled_data
