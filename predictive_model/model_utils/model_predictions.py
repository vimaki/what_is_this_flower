import numpy as np
import torch
import ttach as tta

from .load_dataset import MyDataset


def predict(model, data: MyDataset, on_gpu: bool = True,
            do_tta: bool = True) -> np.ndarray:
    """Make predictions for the transmitted data"""
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
                logits_tta = np.ones((1, list(model.modules())[-1].out_features))
                for transformer in tta.aliases.d4_transform():
                    image_tta = transformer.augment_image(inputs)
                    outputs_tta = model(image_tta).cpu()
                    logits_tta = np.concatenate((logits_tta, outputs_tta), axis=0)
                outputs = np.mean(logits_tta[1:, :], axis=0)
                outputs = torch.from_numpy(outputs)
            else:
                outputs = model(inputs).cpu()
            logits = torch.nn.functional.softmax(outputs, dim=0).data.numpy()
            test_predictions.append(logits)

    probs = np.array(test_predictions)
    return probs


def predict_one_sample(model, inputs: torch.Tensor,
                       on_gpu: bool = True) -> np.ndarray:
    """Make predictions for one image"""
    if on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs
