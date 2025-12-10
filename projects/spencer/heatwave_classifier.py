"""
Heat wave prediction using neural networks.

This script predicts the occurrence of heat waves, defined as 3+ consecutive
days with maximum temperature > 90°F. Features include temperature (min, max,
anomaly) and precipitation from the preceding 2 weeks.

Pedagogical focus: Binary classification with imbalanced data, demonstrating
feature engineering from temporal sequences and model evaluation with precision/recall.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging() -> logging.Logger:
    """Setup timestamped logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"heatwave_classifier_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load temperature and precipitation data from NetCDF file.

    Args:
        filepath: Path to NetCDF file

    Returns:
        temp_min, temp_max, temp_anom, precip arrays (all length N)
    """
    ds = xr.open_dataset(filepath)

    temp_min = ds["temp_min"].values
    temp_max = ds["temp_max"].values
    temp_anom = ds["temp_anom"].values
    precip = ds["precip"].values
    temp_avg = ds["temp_avg"].values

    # Filter out errant 0F values (missing data recorded as 0 instead of NaN)
    n_zeros = (temp_avg == 0).sum()
    if n_zeros > 0:
        logging.warning(f"Found {n_zeros} errant 0°F values in temp_avg, removing them")

    valid_mask = temp_avg != 0

    # Apply mask to all fields
    temp_min = temp_min[valid_mask]
    temp_max = temp_max[valid_mask]
    temp_anom = temp_anom[valid_mask]
    precip = precip[valid_mask]

    # Check for NaN values
    for name, arr in [
        ("temp_min", temp_min),
        ("temp_max", temp_max),
        ("temp_anom", temp_anom),
        ("precip", precip),
    ]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            logging.warning(f"Found {n_nan} NaN values in {name}")

    # Remove any remaining NaN values
    valid_mask = (
        ~np.isnan(temp_min)
        & ~np.isnan(temp_max)
        & ~np.isnan(temp_anom)
        & ~np.isnan(precip)
    )
    temp_min = temp_min[valid_mask]
    temp_max = temp_max[valid_mask]
    temp_anom = temp_anom[valid_mask]
    precip = precip[valid_mask]

    return temp_min, temp_max, temp_anom, precip


def identify_heatwaves(temp_max: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """Identify heat wave days (3+ consecutive days with temp_max > threshold).

    Args:
        temp_max: Array of maximum temperatures
        threshold: Temperature threshold in °F (default: 90)

    Returns:
        Binary array: 1 if day is part of a heat wave, 0 otherwise
    """
    n = len(temp_max)
    is_heatwave = np.zeros(n, dtype=int)

    # Check if each day starts or is part of a 3+ day heat wave
    for i in range(n - 2):
        # Look ahead 3 days
        if np.all(temp_max[i : i + 3] > threshold):
            # Mark all days in this heat wave
            j = i
            while j < n and temp_max[j] > threshold:
                is_heatwave[j] = 1
                j += 1

    return is_heatwave


def create_features_and_labels(
    temp_min: np.ndarray,
    temp_max: np.ndarray,
    temp_anom: np.ndarray,
    precip: np.ndarray,
    heatwave_labels: np.ndarray,
    lookback: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create features from preceding days and labels for heat wave prediction.

    Args:
        temp_min: Minimum temperature array
        temp_max: Maximum temperature array
        temp_anom: Temperature anomaly array
        precip: Precipitation array
        heatwave_labels: Binary heat wave labels
        lookback: Number of preceding days to use as features (default: 14)

    Returns:
        X: Feature array (N-lookback, lookback*4)
        y: Label array (N-lookback,)
    """
    n = len(temp_min)
    n_samples = n - lookback
    n_features = lookback * 4  # 4 variables × lookback days

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Features from days [i, i+lookback)
        X[i, 0:lookback] = temp_min[i : i + lookback]
        X[i, lookback : 2 * lookback] = temp_max[i : i + lookback]
        X[i, 2 * lookback : 3 * lookback] = temp_anom[i : i + lookback]
        X[i, 3 * lookback : 4 * lookback] = precip[i : i + lookback]

        # Label for day i+lookback (the day we're predicting)
        y[i] = heatwave_labels[i + lookback]

    return X, y


def standardize_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Apply z-score standardization using training statistics.

    Args:
        X_train, X_val, X_test: Feature arrays

    Returns:
        Scaled arrays and statistics dictionary
    """
    # Compute statistics from training set only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    # Apply to all splits
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    stats = {"mean": mean, "std": std}

    return X_train_scaled, X_val_scaled, X_test_scaled, stats


class HeatwaveDataset(Dataset):
    """PyTorch Dataset for heat wave classification."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HeatwaveNet(nn.Module):
    """Configurable feedforward network for heat wave classification."""

    def __init__(self, n_features: int, hidden_layers: List[int] = [64, 32]):
        super().__init__()

        if len(hidden_layers) == 0:
            raise ValueError("Must specify at least one hidden layer")

        layers = []

        # Input layer
        layers.append(nn.Linear(n_features, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer (single neuron for binary classification)
        layers.append(nn.Linear(hidden_layers[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    logger: logging.Logger,
    log_interval: int = 5,
) -> Tuple[nn.Module, Dict]:
    """Train the neural network.

    Args:
        model: HeatwaveNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        logger: Logger instance
        log_interval: Log progress every N epochs

    Returns:
        Trained model and training history
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    logger.info("Starting training...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        # Record average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Log progress at specified interval
        if (epoch + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

    logger.info("Training complete")
    return model, history


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline models.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        logger: Logger instance

    Returns:
        Dictionary of baseline results
    """
    results = {}

    # 1. Climatology baseline (predict most common class)
    heatwave_proportion = y_train.mean()
    clim_predictions = (
        np.ones(len(y_test)) if heatwave_proportion > 0.5 else np.zeros(len(y_test))
    )

    clim_acc = accuracy_score(y_test, clim_predictions)
    clim_prec = precision_score(y_test, clim_predictions, zero_division=0)
    clim_rec = recall_score(y_test, clim_predictions, zero_division=0)
    clim_f1 = f1_score(y_test, clim_predictions, zero_division=0)

    results["climatology"] = {
        "Accuracy": clim_acc,
        "Precision": clim_prec,
        "Recall": clim_rec,
        "F1": clim_f1,
    }

    logger.info(
        f"Climatology - Acc: {clim_acc:.3f}, Prec: {clim_prec:.3f}, Rec: {clim_rec:.3f}, F1: {clim_f1:.3f}"
    )

    # 2. Logistic regression baseline
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_probas = lr_model.predict_proba(X_test)[:, 1]

    lr_acc = accuracy_score(y_test, lr_predictions)
    lr_prec = precision_score(y_test, lr_predictions, zero_division=0)
    lr_rec = recall_score(y_test, lr_predictions, zero_division=0)
    lr_f1 = f1_score(y_test, lr_predictions, zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_probas) if len(np.unique(y_test)) > 1 else 0.0

    results["logistic_regression"] = {
        "Accuracy": lr_acc,
        "Precision": lr_prec,
        "Recall": lr_rec,
        "F1": lr_f1,
        "AUC": lr_auc,
    }

    logger.info(
        f"Logistic Regression - Acc: {lr_acc:.3f}, Prec: {lr_prec:.3f}, Rec: {lr_rec:.3f}, F1: {lr_f1:.3f}, AUC: {lr_auc:.3f}"
    )

    return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate neural network on test set.

    Args:
        model: Trained HeatwaveNet
        test_loader: Test data loader
        y_test: Test labels
        logger: Logger instance

    Returns:
        Metrics dictionary, predictions array, and probabilities array
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch)
            all_logits.append(logits.numpy())

    all_logits = np.concatenate(all_logits, axis=0).squeeze()

    # Convert logits to probabilities
    probabilities = 1 / (1 + np.exp(-all_logits))

    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) > 1 else 0.0

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
    }

    logger.info(
        f"Neural Network - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}"
    )

    return metrics, predictions, probabilities


def plot_training_curves(
    history: Dict,
    logger: logging.Logger,
    hidden_layers: List[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    lookback: int,
) -> str:
    """Plot and save training and validation loss curves.

    Args:
        history: Training history dictionary
        logger: Logger instance
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        learning_rate: Learning rate used
        batch_size: Batch size used
        lookback: Lookback period in days

    Returns:
        Path to saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"heatwave_loss_curves_{timestamp}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (BCE)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add hyperparameter info as figure title
    hidden_str = "-".join(map(str, hidden_layers))
    title = (
        f"Training and Validation Loss (Heat Wave Prediction)\n"
        f"Architecture: {hidden_str} | Lookback: {lookback} days | "
        f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size}"
    )
    plt.suptitle(title, fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved loss curves to {plot_path}")
    return plot_path


def plot_evaluation(
    y_test: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    logger: logging.Logger,
    hidden_layers: List[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    lookback: int,
) -> str:
    """Plot confusion matrix and ROC curve.

    Args:
        y_test: True labels
        predictions: Predicted labels
        probabilities: Predicted probabilities
        logger: Logger instance
        hidden_layers: List of hidden layer sizes
        epochs: Number of training epochs
        learning_rate: Learning rate used
        batch_size: Batch size used
        lookback: Lookback period in days

    Returns:
        Path to saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"heatwave_evaluation_{timestamp}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["No Heat Wave", "Heat Wave"],
        yticklabels=["No Heat Wave", "Heat Wave"],
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    # ROC curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        ax2.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax2.set_xlabel("False Positive Rate", fontsize=12)
        ax2.set_ylabel("True Positive Rate", fontsize=12)
        ax2.set_title("ROC Curve", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    # Add hyperparameter info as figure title
    hidden_str = "-".join(map(str, hidden_layers))
    title = (
        f"Heat Wave Prediction Evaluation\n"
        f"Architecture: {hidden_str} | Lookback: {lookback} days | "
        f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size}"
    )
    fig.suptitle(title, fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved evaluation plots to {plot_path}")
    return plot_path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Heat wave prediction using neural networks"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer sizes (default: 64 32)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Log training progress every N epochs (default: 5)",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=14,
        help="Number of preceding days to use as features (default: 14)",
    )

    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    logger = setup_logging()

    # Log configuration
    logger.info("=" * 60)
    logger.info("HEAT WAVE PREDICTION")
    logger.info("=" * 60)
    logger.info(f"Hidden layers: {args.hidden_layers}")
    logger.info(f"Lookback period: {args.lookback} days")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    data_path = "data/central-park-station_daily-data_18690101-20230930.nc"
    temp_min, temp_max, temp_anom, precip = load_data(data_path)

    logger.info(f"Loaded {len(temp_min)} days of data")
    logger.info(f"temp_max range: [{temp_max.min():.2f}, {temp_max.max():.2f}]")

    # Identify heat waves
    logger.info("Identifying heat wave days...")
    heatwave_labels = identify_heatwaves(temp_max, threshold=90.0)
    n_heatwave_days = heatwave_labels.sum()
    heatwave_pct = 100 * n_heatwave_days / len(heatwave_labels)
    logger.info(
        f"Found {n_heatwave_days} heat wave days ({heatwave_pct:.2f}% of total)"
    )

    # Create features and labels
    logger.info(f"Creating features from {args.lookback}-day lookback...")
    X, y = create_features_and_labels(
        temp_min, temp_max, temp_anom, precip, heatwave_labels, lookback=args.lookback
    )
    logger.info(f"Created {len(X)} samples with {X.shape[1]} features each")

    # Train/val/test split (60/20/20)
    logger.info("Splitting data (60% train, 20% val, 20% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    logger.info(
        f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}"
    )
    logger.info(
        f"Train heat wave %: {100*y_train.mean():.2f}%, "
        f"Val: {100*y_val.mean():.2f}%, Test: {100*y_test.mean():.2f}%"
    )

    # Standardize features
    logger.info("Applying z-score standardization...")
    X_train_scaled, X_val_scaled, X_test_scaled, stats = standardize_features(
        X_train, X_val, X_test
    )

    # Create datasets and loaders
    train_dataset = HeatwaveDataset(X_train_scaled, y_train)
    val_dataset = HeatwaveDataset(X_val_scaled, y_val)
    test_dataset = HeatwaveDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate baselines
    logger.info("=" * 60)
    logger.info("BASELINE MODELS")
    logger.info("=" * 60)
    baseline_results = evaluate_baselines(
        X_train_scaled, y_train, X_test_scaled, y_test, logger
    )

    # Train neural network
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK")
    logger.info("=" * 60)
    model = HeatwaveNet(n_features=X.shape[1], hidden_layers=args.hidden_layers)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model architecture: {model}")
    logger.info(f"Total parameters: {n_params}")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.learning_rate,
        logger,
        args.log_interval,
    )

    # Plot training curves
    plot_training_curves(
        history,
        logger,
        args.hidden_layers,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.lookback,
    )

    # Evaluate neural network
    nn_metrics, predictions, probabilities = evaluate_model(
        model, test_loader, y_test, logger
    )

    # Plot evaluation
    plot_evaluation(
        y_test,
        predictions,
        probabilities,
        logger,
        args.hidden_layers,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.lookback,
    )

    # Compare all models
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(
        f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    )
    logger.info("-" * 60)

    for name, metrics in baseline_results.items():
        logger.info(
            f"{name:<20} {metrics['Accuracy']:>10.3f} {metrics['Precision']:>10.3f} "
            f"{metrics['Recall']:>10.3f} {metrics['F1']:>10.3f}"
        )

    logger.info(
        f"{'neural_network':<20} {nn_metrics['Accuracy']:>10.3f} {nn_metrics['Precision']:>10.3f} "
        f"{nn_metrics['Recall']:>10.3f} {nn_metrics['F1']:>10.3f}"
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
