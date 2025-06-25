import torch
import numpy as np

# loader function
from torch.utils.data import DataLoader


from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


class TabularDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def create_dataloaders(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size=32,
    train_shuffle=True,
    test_shuffle=False,
    scale=True,
):
    """
    Create train and test dataloaders from pre-split tabular data.

    Args:
        X_train, y_train: Training features and labels as NumPy arrays
        X_test, y_test: Testing features and labels as NumPy arrays
        batch_size: Batch size for DataLoaders (default=32)
        train_shuffle: Shuffle training data (default=True)
        test_shuffle: Shuffle test data (default=False)
        scale: Whether to standardize features (default=True)

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert to float32 for PyTorch
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    to_tensor = lambda x: torch.tensor(x)

    train_dataset = TabularDataset(X_train, y_train, transform=to_tensor)
    test_dataset = TabularDataset(X_test, y_test, transform=to_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=test_shuffle, drop_last=True
    )

    return train_loader, test_loader


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100  # Percentage accuracy


def mae_fn(y_true, y_pred):
    # Ensure inputs are numpy arrays with at least 1 dimension
    y_true = np.atleast_1d(y_true.numpy())
    y_pred = np.atleast_1d(y_pred.numpy())
    return mean_absolute_error(y_true, y_pred)


def r2_fn(y_true, y_pred):
    return r2_score(y_true.numpy(), y_pred.numpy())


from sklearn.metrics import mean_squared_error


def rmse_fn(y_true, y_pred):
    # Ensure inputs are numpy arrays with at least 1 dimension
    y_true = np.atleast_1d(y_true.numpy())
    y_pred = np.atleast_1d(y_pred.numpy())
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# L1:
def l1_penalty(model):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm


# Training Step
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric_fn,
    device: torch.device = device,
    l1_lambda: float = 0.0,  # L1 regularization strength
):
    train_loss, train_metric = 0.0, 0.0

    model.train()
    model.to(device)

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X).squeeze()
        l1_loss = l1_lambda * l1_penalty(model)

        loss = loss_fn(y_pred, y) + l1_loss  # Add L1 regularization to the loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        train_loss += loss.item()
        train_metric += metric_fn(y.cpu().detach(), y_pred.cpu().detach())

    train_loss /= len(data_loader)
    train_metric /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train Error: {train_metric:.5f}")


# Testing Step
def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    metric_fn,  # e.g., MAE or R²
    device: torch.device = device,
):
    test_loss, test_metric = 0.0, 0.0
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X).squeeze()  # Ensure shape [B], not [B,1]
            loss = loss_fn(test_pred, y)

            test_loss += loss.item()
            test_metric += metric_fn(y.cpu(), test_pred.cpu())

            all_preds.append(test_pred.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        test_loss /= len(data_loader)
        test_metric /= len(data_loader)

        print(f"Test loss: {test_loss:.5f} | Test MAE: {test_metric:.5f}")

    return all_preds, all_labels


def handle_metrics_RM(final_metrics=None, mode="save", filename="models.json"):
    """
    final_metrics: dict with keys:
        'model_name' (str),
        'mae' (float),
        'r2' (float),
        'rmse' (float)
    """
    if mode == "save":
        data = []
        if os.path.exists(filename):
            with open(filename) as f:
                data = json.load(f)
        data.append(final_metrics)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved metrics.")

    elif mode == "display":
        if not os.path.exists(filename):
            print("No metrics saved yet.")
            return
        with open(filename) as f:
            data = json.load(f)

        # Print table header
        print(f"{'Model Name':20} {'MAE':>10} {'R² Score':>10} {'RMSE':>10}")
        print("-" * 55)
        for m in data:
            print(
                f"{m['model_name']:20} "
                f"{m['MAE']:10.4f} "
                f"{m['R2']:10.4f} "
                f"{m['RMSE']:10.4f}"
            )
