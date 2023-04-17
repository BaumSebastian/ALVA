from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# General
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import os

# Own Implementations
from pipeline import generate_samples_with_iterative_epsilons_by_config
from utils import set_random_seed

# Documentation
import matplotlib.pyplot as plt


def generate_samples(cfg):
    # Generate samples
    (
        _,
        generator,
        (z, y, per_z, per_y, epsilons),
    ) = generate_samples_with_iterative_epsilons_by_config(cfg)
    print(f"Generated {len(z)} adversarial samples with generator")
    # Save pictures for later.
    x, per_x = generator(z).detach().cpu(), generator(per_z).detach().cpu()
    return x, y, per_x, per_y, epsilons


def train(train_loader, model, criterion, optimizer, device):
    """
    Function for the training step of the training loop
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


def validate(valid_loader, model, criterion, device):
    """
    Function for the validation step of the training loop
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


def training_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    epochs,
    device,
    print_every=1,
):
    """
    Function defining the entire training loop
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
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)

            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Train accuracy: {100 * train_acc:.2f}\t"
                f"Valid accuracy: {100 * valid_acc:.2f}"
            )

    return (
        model,
        optimizer,
        (train_losses, valid_losses, train_accuracy, valid_accuracy),
    )


def get_accuracy(model, data_loader, device):
    """
    Function for computing the accuracy of the predictions over the entire data_loader
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


def unnormalize_tensor(t):
    return ((t + 1) * 255).type(torch.uint8)


def split_tensor_random(x, y, fraction=1 / 6):
    if fraction > 1:
        raise ValueError("Fraction can't be greater than 1")

    length = len(x)
    split = int(length * fraction)

    random_idx = torch.randperm(len(x))
    return (
        x[random_idx][:split],
        y[random_idx][:split],
        x[random_idx][split:],
        y[random_idx][split:],
    )


def main():
    cfg = load_config("conf", "convergenz_training_100")

    # Set random seed
    set_random_seed(cfg.general.standard.random_seed)

    # Check if folders exist and copy original values
    for directory in cfg.save_paths.values():
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get Device
    DEVICE = torch.device(cfg.general.device)

    # Set Hyperparameters
    BATCH_SIZE = cfg.hyperparameter.BATCH_SIZE
    LEARNING_RATE = cfg.hyperparameter.LEARNING_RATE
    N_EPOCHS = cfg.hyperparameter.N_EPOCHS
    N_EPSILON_EPOCHS = cfg.n_conv_epochs

    # General information about mnist
    DATA_DIM = tuple(cfg.dataset.shape.values())
    CLASSES = cfg.dataset.classes

    # Load Data
    original_training_data, _ = mnist.get_dataset(cfg.dataset.general.offset)
    per_training_data = mnist.PerturbatedMnist(
        cfg.dataset.path, "training", transform=mnist.get_standard_transformation()
    )
    per_test_data = mnist.PerturbatedMnist(
        cfg.dataset.path, "test", transform=mnist.get_standard_transformation()
    )

    perturbated_percentage = []
    for epsilon_epoch in range(N_EPSILON_EPOCHS):
        # Declare experiment specific variables
        run_name = f"{epsilon_epoch:03}"
        losses_path = f"{cfg.save_paths.losses}/loss_{run_name}.pt"
        epsilons_path = f"{cfg.save_paths.epsilons}/epsilons_{run_name}.pt"
        model_path = f"{cfg.save_paths.models}/lenet_{run_name}.pth"

        # Reload data and create Dataloader
        per_training_data.load_data()
        per_test_data.load_data()
        train_loader = DataLoader(
            per_training_data, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(per_test_data, batch_size=BATCH_SIZE, shuffle=True)

        perturbated_percentage.append(
            (
                per_training_data.get_perturbated_percentage(),
                per_test_data.get_perturbated_percentage(),
            )
        )

        # Start run
        print(f"\n--------------------------")
        print(f"Executing Experiment: #{run_name}")
        print(
            f"\nPerturbated images: {round(per_training_data.get_perturbated_percentage(), 3)}%"
        )
        print(f"--------------------------\n")

        # Reinitialize model
        model = LeNet5().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Train the model on the dataset
        print("\nTRAINING\n")
        model, optimizer, metrics = training_loop(
            model, criterion, optimizer, train_loader, test_loader, N_EPOCHS, DEVICE
        )

        # Save the model and load it into the config
        torch.save(model, model_path)
        cfg.models.classifier = model_path

        # Exceut the pipelin to generate samples for every class
        per_xs = []
        targets = []
        all_target_figures = []
        epsilons = []

        print("\nGENERATING\n")
        for target in CLASSES:
            add_config_entry(cfg, "target", target)
            print("\nGoing for " + str(target) + "\n")

            # Generating images and storing figures
            x, y, per_x, per_y, epsilon = generate_samples(cfg)
            figures = generate_plots(
                x,
                y,
                per_x,
                per_y,
                cfg.target,
                original_training_data,
                DATA_DIM,
                cfg.dataset.name,
            )

            per_xs.append(unnormalize_tensor(per_x))
            targets.append(torch.full((1, per_x.shape[0]), target, dtype=int))
            all_target_figures.append((target, figures))
            epsilons.append((target, epsilon))

        # Process data
        x_test, y_test, x_train, y_train = split_tensor_random(
            torch.cat(per_xs).view(-1, 28, 28), torch.cat(targets, dim=1).view(-1)
        )

        # Save torchs
        torch.save(torch.Tensor(metrics), losses_path)
        torch.save(epsilons, epsilons_path)
        torch.save((x_train, y_train), f"{cfg.save_paths.runs}/{run_name}_training.pt")
        torch.save((x_test, y_test), f"{cfg.save_paths.runs}/{run_name}_test.pt")
        torch.save(
            torch.Tensor(perturbated_percentage),
            cfg.save_paths.general + "/perturbated_percentage.pt",
        )
        # Save figures
        # Save figures
        pic_folder = f"{cfg.save_paths.pictures}/{run_name}"
        if not os.path.isdir(pic_folder):
            os.mkdir(pic_folder)
        for name, figures in all_target_figures:
            for idx, fig in enumerate(figures):
                if fig != None:
                    fig.savefig(f"{pic_folder}/{idx}_{name}.png")
                    fig.savefig(f"{pic_folder}/{idx}_{name}.svg")
        plt.close("all")

        print(f"Saved {x_test.shape[0] + x_train.shape[0]} values")


if __name__ == "__main__":
    main()
