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
