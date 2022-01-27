import json
import sys
from typing import Tuple

from torch import device, load, Tensor
from torch.nn import Linear
from torchvision import models
from torchvision import transforms

sys.path.insert(1, '../predictive_model/model_utils')

from load_dataset import MyDataset

PATH_TO_MODEL = '../predictive_model/model_effnet_b5_full_weights.pth'
LABEL_ENCODER = '../predictive_model/label_encoder.json'
FLOWER_DICTIONARY = '../scraping_dataset/flower_types.json'

with open(LABEL_ENCODER) as f:
    label_encoder = json.load(f)

with open(FLOWER_DICTIONARY) as f:
    flower_dict = json.load(f)

model = models.efficientnet_b5()
model.classifier[1] = Linear(model.classifier[1].in_features, len(flower_dict))
model.load_state_dict(load(PATH_TO_MODEL, map_location=device('cpu'))['model_state_dict'])
model.eval()


def transform_image(image_path: str, rescale_size: int = 224) -> Tensor:
    # Set image transformations for transfer to the model input
    transform = transforms.Compose([
        transforms.Resize((rescale_size, rescale_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = MyDataset.load_sample(image_path)
    image = MyDataset.crop_image(image)
    image = transform(image).unsqueeze(0)
    return image


def get_inference(image_path: str) -> Tuple[str, str]:
    image = transform_image(image_path)
    outputs = model.forward(image)
    _, y_hat = outputs.max(1)
    predicted_label = str(y_hat.item())

    flower_name_eng = label_encoder[predicted_label]
    flower_name_rus = flower_dict[flower_name_eng]
    return flower_name_eng, flower_name_rus
