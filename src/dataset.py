import os

import PIL
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

LABELS_TEXT2INT = {"real": 0, "fake": 1}
LABELS_INT2TEXT = {0: "real", 1: "fake"}


def split_dataframe(df, validation_size, test_size=None, seed=None):
    dfs = {}

    if test_size is None:
        test_size = 0

    dfs["train"], temp_df = train_test_split(
        df,
        test_size=test_size + validation_size,
        stratify=df["label"].values,
        random_state=seed,
    )

    if test_size:
        dfs["valid"], dfs["test"] = train_test_split(
            temp_df,
            test_size=test_size / (test_size + validation_size),
            stratify=temp_df["label"].values,
            random_state=seed,
        )
    else:
        dfs["valid"] = temp_df

    return dfs


def create_datasets(folder_paths=None, transformations=None):
    return {
        k: ImageDataset(folder_paths[k], transform=transformations[k])
        for k in folder_paths.keys()
    }


def create_dataloaders(datasets, train_batch_size, eval_batch_size):
    num_workers = os.cpu_count()
    dataloaders = {}

    for k in datasets.keys():
        if k == "train":
            dataloaders[k] = torch.utils.data.DataLoader(
                datasets[k],
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            dataloaders[k] = torch.utils.data.DataLoader(
                datasets[k],
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

    return dataloaders


def create_transforms(keys, input_size, mean, std):
    return {
        k: T.Compose(
            [
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        if k == "train"
        else T.Compose(
            [
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        for k in keys
    }


def create_single_batch_dataloader(full_ds, size=8):
    idx = torch.randint(len(full_ds), (size,))
    batch_ds = torch.utils.data.Subset(full_ds, idx)

    return torch.utils.data.DataLoader(
        batch_ds,
        batch_size=size,
        shuffle=True,
    )


class ImageDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        idx = self.imgs[index][0].split("/")[-1]
        return idx, sample, target

    def find_classes(self, directory):
        return (LABELS_TEXT2INT.keys(), LABELS_TEXT2INT)
