# utils.py
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.data import create_dataset, create_loader
from timm.loss import LabelSmoothingCrossEntropy
import matplotlib.pyplot as plt

# Constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_PCT = 0.9
data_dir = "/content/data1"
input_size = (3, 128, 128)
batch_size = 32
img_size = 128
num_classes = 2
interpolation = 'bicubic'

def prepare_data():
    data_transforms = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    image_datasets = datasets.ImageFolder(data_dir, data_transforms)

    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes
    print(dataset_sizes, class_names)

    val_size = int(dataset_sizes * 0.15)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(image_datasets,
                                                                             [dataset_sizes - 2 * val_size, val_size,
                                                                              val_size],
                                                                             generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, test_dataset, class_names

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_DEFAULT_MEAN)
    std = np.array(IMAGENET_DEFAULT_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def eval_corona(model, loader_val, device, val_len):
    model.eval()

    preds = []
    with torch.no_grad():
        for x, t in loader_val:
            x, t = x.to(device), t.to(device)
            logits = model(x)
            preds.append(torch.sum(torch.max(logits, dim=1)[1] == t))

    return sum(preds).item() / val_len
