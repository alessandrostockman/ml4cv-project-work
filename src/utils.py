import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset import LABELS_INT2TEXT


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_images(samples, labels=None, mean=(0, 0, 0), std=(1, 1, 1)):
    if type(mean) != torch.Tensor:
        mean = torch.tensor(mean)
    if type(std) != torch.Tensor:
        std = torch.tensor(std)

    mean = mean[:, None, None]
    std = std[:, None, None]

    if type(samples[0]) == str:
        transform = T.Compose([T.ToTensor()])
        samples = {
            "image": [transform(PIL.Image.open(img_path)) for img_path in samples]
        }
    elif type(samples[0]) == torch.Tensor:
        samples = {"image": samples[0], "label": samples[1]}
    elif type(samples[0]) == list:
        samples = {
            "image": torch.cat([x.unsqueeze(dim=0) for x in samples[0]]),
            "label": samples[1],
        }

    if labels is not None:
        samples["label"] = labels

    imgs = [(img * std) + mean for img in samples["image"]]
    grid_img = torchvision.utils.make_grid(imgs)
    plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))

    if "label" in samples:
        if type(samples["label"]) == torch.Tensor:
            samples["label"] = samples["label"].tolist()
        plt.title(", ".join([LABELS_INT2TEXT[label] for label in samples["label"]]))

    plt.show()


def get_mean_and_std(dataset, batch_size):
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1
    )
    sums = torch.zeros(3)
    sums_of_square = torch.zeros(3)
    count = 0

    for _, images, _ in tqdm(dl):
        b, _, h, w = images.shape
        num_pix_in_batch = b * h * w
        sums += torch.sum(images, dim=[0, 2, 3])
        sums_of_square += torch.sum(images**2, dim=[0, 2, 3])
        count += num_pix_in_batch

    mean = sums / count
    var = (sums_of_square / count) - (mean**2)
    std = torch.sqrt(var)

    return mean, std
