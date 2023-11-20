import re
import os
import csv
import PIL
import torch
import numpy as np
from torchvision import transforms, datasets as tvds
from torch.utils.data import DataLoader, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from collections import namedtuple

CSV = namedtuple("CSV", ["header", "index", "data"])
DATASET_DICT = dict()
DATASET_INFO = dict()

def register_dataset(cls):
    name = cls.__name__.lower()
    DATASET_DICT[name] = cls
    info = dict()
    for k, v in cls.__dict__.items():
        if re.match(r"__\w+__", k) is None and not callable(v):
            info[k] = v
    DATASET_INFO[name] = info
    return cls


@register_dataset
class CIFAR10(tvds.CIFAR10):
    resolution = (32, 32)
    channels = 3
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    _transform = transforms.PILToTensor()
    train_size = 50000
    test_size = 10000

def train_val_split(dataset, val_size, random_seed=None):
    train_size = DATASET_INFO[dataset]["train_size"]
    if random_seed is not None:
        np.random.seed(random_seed)
    train_inds = np.arange(train_size)
    np.random.shuffle(train_inds)
    val_size = int(train_size * val_size)
    val_inds, train_inds = train_inds[:val_size], train_inds[val_size:]
    return train_inds, val_inds


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):  # noqa
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(
        dataset,
        batch_size,
        split,
        val_size=0.1,
        random_seed=None,
        root=None,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        distributed=False
):
    assert isinstance(val_size, float) and 0 <= val_size < 1

    name, dataset = dataset, DATASET_DICT[dataset]
    transform = dataset.transform
    if distributed:
        batch_size = batch_size // int(os.environ.get("WORLD_SIZE", "1"))
    data_kwargs = {"root": root, "transform": transform}
    if name == "celeba":
        data_kwargs["split"] = split
    elif name in {"mnist", "cifar10"}:
        data_kwargs["download"] = True
        data_kwargs["train"] = split != "test"
    dataset = dataset(**data_kwargs)

    if data_kwargs.get("train", False) and val_size > 0.:
        train_inds, val_inds = train_val_split(name, val_size, random_seed)
        dataset = Subset(dataset, {"train": train_inds, "valid": val_inds}[split])

    dataloader_configs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers
    }
    dataloader_configs["sampler"] = sampler = DistributedSampler(
        dataset, shuffle=True, seed=random_seed, drop_last=drop_last) if distributed else None
    dataloader_configs["shuffle"] = (sampler is None) if split in {"train", "all"} else False
    dataloader = DataLoader(dataset, **dataloader_configs)
    return dataloader, sampler
