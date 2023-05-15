from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# For annotation
from typing import Tuple

# Imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_standard_transformation() -> transforms.transforms.Compose:
    """
    The standard transformation applied to mnist dataset.

    :return: Returns the standard transformation, when getting MNIST dataset.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_dataset(
    root: str = ".", transform=None, download=False
) -> Tuple(Dataset, Dataset):
    """
    Loads the MNIST Datasets. If not downloaded it downloads the dataset.

    :param root: The root directory to the data directory. Default: ".".
    :param transform: The transformation applied to MNIST.
    :param download: Downloads the dataset.
    :return: The test and train dataset normalized with -1 and 1 as tensor.
    """
    training_data = datasets.MNIST(
        root=root, train=True, download=download, transform=transform
    )

    test_data = datasets.MNIST(
        root=root, train=False, download=download, transform=transform
    )

    return (training_data, test_data)


def get_dataloader(
    root: str, batch_size: int, target=None, shuffle: bool = True
) -> Tuple(DataLoader, DataLoader):
    """
    Loads the MNIST DataLoader.

    :param root: The relative directory to the data directory. root/data.
    :param batch_size: the batch size for dataloader.
    :param shuffle: if true, data gets shuffled.
    :param target: if target not None, returns just dataloader with target equal to parameter

    :return: The test and train DataLoader.
    """

    training_data, test_data = (
        get_dataset(root)
        if target != None
        else get_dataset_subset_by_target(root, target)
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader


def get_dataset_subset_by_target(root, target) -> Tuple(Subset, Subset):
    """
    Gets a subset of dataset

    :param root: The relative directory to the data directory. root/data.
    :param target: The target to select subset

    :return: two subsets with specific target
    """
    training_data, test_data = get_dataset(root)
    training_mask, test_mask = (training_data.targets == target).nonzero(as_tuple=True)[
        0
    ], (test_data.targets == target).nonzero(as_tuple=True)[0]

    return Subset(training_data, training_mask), Subset(test_data, test_mask)
