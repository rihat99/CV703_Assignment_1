import torch

from tqdm import tqdm

from utils import save_model

def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss for the epoch.
    """

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ascii=True)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        correct += (output.argmax(dim=1) == target).sum().item()
        total += target.size(0)

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy

def val_step(
        model: torch.nn.Module,
        val_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss and accuracy for the val set.
    """

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, ascii=True):
            data, target = data.to(device), target.to(device)

            output = model(data)

            val_loss += loss_fn(output, target).item()

            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)

    accuracy = correct / total

    return val_loss, accuracy


def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: torch.device,
        epochs: int,
        save_dir: str,
        linear_probing_epochs=None
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_loss = 1e10

    for epoch in range(1, epochs + 1):
        if linear_probing_epochs is not None:
            if epoch == linear_probing_epochs:
                for param in model.parameters():
                    param.requires_grad = True


        print(f"Epoch {epoch}:")
        train_loss, train_accuracy = train_step(model, train_loader, loss_fn, optimizer,  device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {(train_accuracy * 100):.4f}")

        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        results["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if lr_scheduler is not None:
            lr_scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)

        val_loss, val_accuracy = val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {(val_accuracy * 100):.4f}")
        print()

        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir + "/best_model.pth")

        save_model(model, save_dir + "/last_model.pth")


    return results