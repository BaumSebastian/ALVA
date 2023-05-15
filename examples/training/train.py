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
from .validate import validate
import datetime
from torch.utils.data import DataLoader


def train(
    train_loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer: torch.optim,
    device: torch.device,
) -> Tuple(torch.nn.Module, torch.optim, float):
    """
    Function for the training step of the training loop.

    :param train_loader: dataloader with training data.
    :param model: the model that will be validated.
    :param criterion: the criterion to calculate the loss.
    :param optimizer: the optimizer to optimize the model.
    :param device:  The device to put the data and networks on.

    :return: The model, the optimizer and the epoch loss.
    """
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()
        X, y_true = X.to(device), y_true.to(device)

        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def training_loop(
    model: torch.nn.Module,
    criterion: torch.nn,
    optimizer: torch.optim,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: torch.device,
    print_every: int = 1,
) -> Tuple(torch.nn.Module, torch.optim, Tuple(float, float, float, float)):
    """
    Function defining the entire training loop.

    :param model: the model that will be validated.
    :param criterion: the criterion to calculate the loss.
    :param optimizer: the optimizer to optimize the model.
    :param train_loader: dataloader with training data.
    :param valid_loader: dataloader with validation data.
    :param epochs: The number of epochs for training
    :param device:  The device to put the data and networks on.
    :param print_every: frequency of printing information.

    :return: The model, the optimizer, Tuple of train-/validation-loss and train-/validation-accuracy.
    """
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []

    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            # Calculate accuracy
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)

            __print_training_information(
                epoch, train_loss, valid_loss, train_acc, valid_acc
            )

    return (
        model,
        optimizer,
        (train_losses, valid_losses, train_accuracy, valid_accuracy),
    )


def __print_training_information(
    epoch: int, train_loss: float, valid_loss: float, train_acc: float, valid_acc: float
) -> None:
    """
    Prints the information about the training process.

    :param epoch: The epoch of the training.
    :param train_loss: The loss of the training.
    :param valid_loss: The loss of the validation.
    :param train_acc: The accuracy of the training.
    :param valid_acc: The accuracy of the training.

    :return: None
    """
    print(
        f"{datetime.datetime.now().time().replace(microsecond=0)} --- "
        f"Epoch: {epoch}\t"
        f"Train loss: {train_loss:.4f}\t"
        f"Valid loss: {valid_loss:.4f}\t"
        f"Train accuracy: {100 * train_acc:.2f}\t"
        f"Valid accuracy: {100 * valid_acc:.2f}"
    )


def get_accuracy(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """
    Function for computing the accuracy of the predictions over the entire data_loader.

    :param model: the model that will be validated.
    :param data_loader: dataloader with data.
    :param device: The device to put the data and networks on.

    :return: accuracy.
    """

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X, y_true = X.to(device), y_true.to(device)

            y_hat = model(X)
            predicted_labels = y_hat.max(1, keepdim=True)[1].squeeze()
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).float().sum()

    return correct_pred.float() / n
