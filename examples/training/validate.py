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
