from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# For annotation
from typing import Tuple

# Imports
import torch
from torch.utils.data import DataLoader


def validate(
    valid_loader: DataLoader, model: torch.nn.Module, criterion, device: torch.device
) -> Tuple(torch.nn.Module, float):
    """
    Function for the validation step of the training loop

    :param valid_loader: dataloader with validation data
    :param model: the model that will be validated
    :param criterion: the criterion to calculate the loss
    :param device: the device to load the data

    :param return: the model and the epoch loss
    """
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X, y_true = X.to(device), y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss
