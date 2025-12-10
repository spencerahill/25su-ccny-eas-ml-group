"""
Next-day temperature prediction using neural networks.

This script predicts tomorrow's average temperature from today's average temperature.
It supports both raw temperatures and temperature anomalies, demonstrating the impact
of problem formulation on model performance.

Pedagogical focus: A cleaner regression example than precipitation prediction,
with stronger signal (R² > 0.7) that helps students see meaningful ML performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
from typing import Tuple, Dict, List


def setup_logging() -> logging.Logger:
    """Setup timestamped logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"temp_regression_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def load_data(
    filepath: str, use_anomalies: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Load temperature data from NetCDF file.

    Args:
        filepath: Path to NetCDF file
        use_anomalies: If True, use temp_anom; if False, use temp_avg

    Returns:
        temperature: Temperature array (N,)
        time_coord: Time coordinate array (N,)
    """
    ds = xr.open_dataset(filepath)

    field_name = "temp_anom" if use_anomalies else "temp_avg"
    temperature = ds[field_name].values
    time_coord = ds["time"].values

    # Check for NaN values
    n_nan = np.isnan(temperature).sum()
    if n_nan > 0:
        logging.warning(f"Found {n_nan} NaN values in {field_name}, removing them")
        valid_mask = ~np.isnan(temperature)
        temperature = temperature[valid_mask]
        time_coord = time_coord[valid_mask]

    return temperature, time_coord


def create_next_day_pairs(
    temperature: np.ndarray, time_coord: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create (X_today, y_tomorrow) pairs for next-day prediction.

    Args:
        temperature: Temperature array of length N
        time_coord: Time coordinate array of length N

    Returns:
        X: Today's temperatures (N-1,)
        y: Tomorrow's temperatures (N-1,)
        time_X: Time coordinates aligned with X (N-1,)
    """
    X = temperature[:-1]  # All days except last
    y = temperature[1:]  # All days except first
    time_X = time_coord[:-1]  # Align time with X

    return X, y, time_X


def compute_climatology(
    y_train: np.ndarray, time_train: np.ndarray
) -> Dict[int, float]:
    """Compute day-of-year climatology from training set.

    Args:
        y_train: Training target values
        time_train: Training time coordinates

    Returns:
        Dictionary mapping day-of-year (1-366) to mean temperature
    """
    climatology = {}

    # Convert numpy datetime64 to day-of-year
    days_of_year = (
        time_train.astype("datetime64[D]").astype("datetime64[M]").astype(int) % 12 + 1
    )

    # Use pandas for easier day-of-year extraction
    import pandas as pd

    time_df = pd.to_datetime(time_train)
    days_of_year = time_df.dayofyear.values

    # Compute mean for each day-of-year
    for doy in range(1, 367):
        mask = days_of_year == doy
        if mask.sum() > 0:
            climatology[doy] = y_train[mask].mean()
        else:
            # If no data for this day (e.g., Feb 29 in non-leap years), use overall mean
            climatology[doy] = y_train.mean()

    return climatology


def standardize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict
]:
    """Apply z-score standardization using training statistics.

    Args:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Target arrays

    Returns:
        Scaled arrays and statistics dictionaries
    """
    # Compute statistics from training set only
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = y_train.mean()
    y_std = y_train.std()

    # Apply to all splits
    X_train_scaled = (X_train - x_mean) / x_std
    X_val_scaled = (X_val - x_mean) / x_std
    X_test_scaled = (X_test - x_mean) / x_std

    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    x_stats = {"mean": x_mean, "std": x_std}
    y_stats = {"mean": y_mean, "std": y_std}

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        x_stats,
        y_stats,
    )


def inverse_transform(y_scaled: np.ndarray, y_stats: Dict) -> np.ndarray:
    """Convert predictions back to original scale.

    Args:
        y_scaled: Scaled predictions
        y_stats: Dictionary with 'mean' and 'std'

    Returns:
        Predictions on original scale
    """
    return y_scaled * y_stats["std"] + y_stats["mean"]


class TempDataset(Dataset):
    """PyTorch Dataset for temperature regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (N, 1)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TempNet(nn.Module):
    """Configurable feedforward network for temperature regression."""

    def __init__(self, hidden_layers: List[int] = [32]):
        super().__init__()

        if len(hidden_layers) == 0:
            raise ValueError("Must specify at least one hidden layer")

        layers = []

        # Input layer (1 input -> first hidden)
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer (last hidden -> 1 output)
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
) -> Tuple[nn.Module, Dict]:
    """Train the neural network.

    Args:
        model: TempNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        logger: Logger instance

    Returns:
        Trained model and training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    logger.info("Starting training...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        # Record average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
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
    time_train: np.ndarray,
    time_test: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline models.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        time_train, time_test: Time coordinates
        logger: Logger instance

    Returns:
        Dictionary of baseline results
    """
    results = {}

    # 1. Persistence baseline (tomorrow = today)
    persistence_predictions = X_test
    persistence_mae = mean_absolute_error(y_test, persistence_predictions)
    persistence_rmse = np.sqrt(mean_squared_error(y_test, persistence_predictions))
    persistence_r2 = r2_score(y_test, persistence_predictions)

    results["persistence"] = {
        "MAE": persistence_mae,
        "RMSE": persistence_rmse,
        "R²": persistence_r2,
    }

    logger.info(
        f"Persistence - MAE: {persistence_mae:.3f}, RMSE: {persistence_rmse:.3f}, R²: {persistence_r2:.3f}"
    )

    # 2. Climatology baseline (day-of-year mean)
    climatology = compute_climatology(y_train, time_train)

    import pandas as pd

    test_doy = pd.to_datetime(time_test).dayofyear.values
    clim_predictions = np.array([climatology[doy] for doy in test_doy])

    clim_mae = mean_absolute_error(y_test, clim_predictions)
    clim_rmse = np.sqrt(mean_squared_error(y_test, clim_predictions))
    clim_r2 = r2_score(y_test, clim_predictions)

    results["climatology"] = {"MAE": clim_mae, "RMSE": clim_rmse, "R²": clim_r2}

    logger.info(
        f"Climatology - MAE: {clim_mae:.3f}, RMSE: {clim_rmse:.3f}, R²: {clim_r2:.3f}"
    )

    # 3. Linear regression baseline
    lr_model = LinearRegression()
    lr_model.fit(X_train.reshape(-1, 1), y_train)
    lr_predictions = lr_model.predict(X_test.reshape(-1, 1))

    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
    lr_r2 = r2_score(y_test, lr_predictions)

    results["linear_regression"] = {"MAE": lr_mae, "RMSE": lr_rmse, "R²": lr_r2}

    logger.info(
        f"Linear Regression - MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}, R²: {lr_r2:.3f}"
    )

    return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    y_test: np.ndarray,
    y_stats: Dict,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate neural network on test set.

    Args:
        model: Trained TempNet
        test_loader: Test data loader
        y_test: Test targets (original scale)
        y_stats: Target statistics for inverse transform
        logger: Logger instance

    Returns:
        Metrics dictionary and predictions array
    """
    model.eval()
    predictions_scaled = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = model(X_batch)
            predictions_scaled.append(pred.numpy())

    predictions_scaled = np.concatenate(predictions_scaled, axis=0).squeeze()

    # Inverse transform to original scale
    predictions = inverse_transform(predictions_scaled, y_stats)

    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2}

    logger.info(f"Neural Network - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

    return metrics, predictions


def plot_training_curves(history: Dict, logger: logging.Logger) -> str:
    """Plot and save training and validation loss curves.

    Args:
        history: Training history dictionary
        logger: Logger instance

    Returns:
        Path to saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"temp_loss_curves_{timestamp}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved loss curves to {plot_path}")
    return plot_path


def plot_predictions(
    y_test: np.ndarray, predictions: np.ndarray, logger: logging.Logger
) -> str:
    """Plot predicted vs actual temperatures with residuals.

    Args:
        y_test: Actual temperatures
        predictions: Predicted temperatures
        logger: Logger instance

    Returns:
        Path to saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"temp_predictions_{timestamp}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: predicted vs actual
    ax1.scatter(y_test, predictions, alpha=0.3, s=10)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax1.set_xlabel("Actual Temperature", fontsize=12)
    ax1.set_ylabel("Predicted Temperature", fontsize=12)
    ax1.set_title("Predicted vs Actual Temperature", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Residual plot
    residuals = predictions - y_test
    ax2.scatter(predictions, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted Temperature", fontsize=12)
    ax2.set_ylabel("Residual (Predicted - Actual)", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Saved prediction plots to {plot_path}")
    return plot_path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Next-day temperature prediction using neural networks"
    )

    parser.add_argument(
        "--use-anomalies",
        action="store_true",
        help="Use temperature anomalies instead of raw temperatures",
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
        default=[32],
        help="Hidden layer sizes (default: 32)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    logger = setup_logging()

    # Log configuration
    temp_type = "anomalies" if args.use_anomalies else "raw temperatures"
    logger.info("=" * 60)
    logger.info("NEXT-DAY TEMPERATURE PREDICTION")
    logger.info("=" * 60)
    logger.info(f"Temperature type: {temp_type}")
    logger.info(f"Hidden layers: {args.hidden_layers}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    data_path = "data/central-park-station_daily-data_18690101-20230930.nc"
    temperature, time_coord = load_data(data_path, use_anomalies=args.use_anomalies)

    logger.info(f"Loaded {len(temperature)} days of data")
    logger.info(
        f"Temperature range: [{temperature.min():.2f}, {temperature.max():.2f}]"
    )
    logger.info(
        f"Temperature mean: {temperature.mean():.2f}, std: {temperature.std():.2f}"
    )

    # Create next-day pairs
    logger.info("Creating next-day prediction pairs...")
    X, y, time_X = create_next_day_pairs(temperature, time_coord)
    logger.info(f"Created {len(X)} (today, tomorrow) pairs")

    # Train/val/test split (60/20/20)
    logger.info("Splitting data (60% train, 20% val, 20% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # Also split time for climatology baseline
    time_temp, time_test = train_test_split(time_X, test_size=0.2, random_state=42)
    time_train, time_val = train_test_split(time_temp, test_size=0.25, random_state=42)

    logger.info(
        f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    # Standardize
    logger.info("Applying z-score standardization...")
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        x_stats,
        y_stats,
    ) = standardize_data(X_train, X_val, X_test, y_train, y_val, y_test)

    logger.info(
        f"X statistics - mean: {x_stats['mean']:.3f}, std: {x_stats['std']:.3f}"
    )
    logger.info(
        f"y statistics - mean: {y_stats['mean']:.3f}, std: {y_stats['std']:.3f}"
    )

    # Create datasets and loaders
    train_dataset = TempDataset(X_train_scaled, y_train_scaled)
    val_dataset = TempDataset(X_val_scaled, y_val_scaled)
    test_dataset = TempDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate baselines
    logger.info("=" * 60)
    logger.info("BASELINE MODELS")
    logger.info("=" * 60)
    baseline_results = evaluate_baselines(
        X_train, y_train, X_test, y_test, time_train, time_test, logger
    )

    # Train neural network
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK")
    logger.info("=" * 60)
    model = TempNet(hidden_layers=args.hidden_layers)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model architecture: {model}")
    logger.info(f"Total parameters: {n_params}")

    model, history = train_model(
        model, train_loader, val_loader, args.epochs, args.learning_rate, logger
    )

    # Plot training curves
    plot_training_curves(history, logger)

    # Evaluate neural network
    nn_metrics, predictions = evaluate_model(
        model, test_loader, y_test, y_stats, logger
    )

    # Plot predictions
    plot_predictions(y_test, predictions, logger)

    # Compare all models
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    logger.info("-" * 60)

    for name, metrics in baseline_results.items():
        logger.info(
            f"{name:<20} {metrics['MAE']:>10.3f} {metrics['RMSE']:>10.3f} {metrics['R²']:>10.3f}"
        )

    logger.info(
        f"{'neural_network':<20} {nn_metrics['MAE']:>10.3f} {nn_metrics['RMSE']:>10.3f} {nn_metrics['R²']:>10.3f}"
    )

    # Determine if NN beat baselines
    logger.info("=" * 60)
    persistence_r2 = baseline_results["persistence"]["R²"]
    lr_r2 = baseline_results["linear_regression"]["R²"]
    nn_r2 = nn_metrics["R²"]

    if nn_r2 > max(persistence_r2, lr_r2):
        logger.info("Neural network OUTPERFORMS all baselines")
    elif nn_r2 > persistence_r2:
        logger.info("Neural network beats persistence but not linear regression")
    else:
        logger.info("Neural network does NOT beat persistence baseline")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
