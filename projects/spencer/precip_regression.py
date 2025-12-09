"""
NYC Precipitation Regression Model

Predicts same-day precipitation from temperature variables and lagged features.
Uses a simple feedforward neural network and compares against baseline models.

Predictors:
- Same day: temp_min, temp_max
- Previous N days: temp_min, temp_max, precip (N configurable, default=7)

Baselines:
- Climatology: Mean precipitation for each day-of-year
- Persistence: Previous day's precipitation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
from typing import Tuple, Dict


def setup_logging() -> None:
    """Set up logging with both console and file handlers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_filename = f"{log_directory}/precip_regression_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )

    # Set console handler to INFO, file handler to DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Suppress noisy matplotlib messages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_filename}")


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, xr.DataArray]:
    """Load precipitation and temperature data from netCDF file.

    Returns:
        Tuple of (precip, temp_min, temp_max, time_coord)
    """
    with xr.open_dataset(filepath) as ds:
        precip = ds["precip"].values
        temp_min = ds["temp_min"].values
        temp_max = ds["temp_max"].values
        time_coord = ds["time"]

    logging.info(f"Loaded {len(precip)} daily records")
    logging.info(
        f"Date range: {time_coord[0].dt.strftime('%Y-%m-%d').values} to "
        f"{time_coord[-1].dt.strftime('%Y-%m-%d').values}"
    )
    logging.debug(f"Data shapes: precip={precip.shape}, temp_min={temp_min.shape}")

    return precip, temp_min, temp_max, time_coord


def create_lagged_features(
    precip: np.ndarray,
    temp_min: np.ndarray,
    temp_max: np.ndarray,
    time_coord: xr.DataArray,
    n_lags: int = 7,
    min_precip_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """Create feature matrix with lagged variables.

    Features for each day:
    - Same day: temp_min, temp_max
    - Previous n_lags days: temp_min, temp_max, precip

    Total features: 2 + n_lags * 3 = 2 + 3*n_lags

    Args:
        precip: Daily precipitation (target)
        temp_min: Daily minimum temperature
        temp_max: Daily maximum temperature
        time_coord: Time coordinate
        n_lags: Number of previous days to include (default 7)
        min_precip_threshold: Only include days with precip >= threshold (default 0.0 keeps all)

    Returns:
        Tuple of (features, target, time_coord_truncated)
        - features shape: (n_samples, n_features)
        - target shape: (n_samples,)
    """
    # Remove rows with any NaN values
    valid_mask = ~(np.isnan(precip) | np.isnan(temp_min) | np.isnan(temp_max))
    precip_clean = precip[valid_mask]
    temp_min_clean = temp_min[valid_mask]
    temp_max_clean = temp_max[valid_mask]
    time_clean = time_coord[valid_mask]

    n_samples = len(precip_clean)
    n_features = 2 + n_lags * 3  # same-day temps + lagged (temps + precip)

    # We lose n_lags days at the start since we need lookback
    n_valid_samples = n_samples - n_lags

    # Initialize feature matrix for ALL valid days first
    X_all = np.zeros((n_valid_samples, n_features))
    y_all = np.zeros(n_valid_samples)
    time_all = []

    # For each valid day (starting from index n_lags)
    for i in range(n_lags, n_samples):
        row_idx = i - n_lags
        feature_idx = 0

        # Same-day temperature features
        X_all[row_idx, feature_idx] = temp_min_clean[i]
        X_all[row_idx, feature_idx + 1] = temp_max_clean[i]
        feature_idx += 2

        # Lagged features (going backwards in time)
        for lag in range(1, n_lags + 1):
            lag_idx = i - lag
            X_all[row_idx, feature_idx] = temp_min_clean[lag_idx]
            X_all[row_idx, feature_idx + 1] = temp_max_clean[lag_idx]
            X_all[row_idx, feature_idx + 2] = precip_clean[lag_idx]
            feature_idx += 3

        # Target is same-day precipitation
        y_all[row_idx] = precip_clean[i]
        time_all.append(time_clean[i].values)

    # Filter based on precipitation threshold
    keep_mask = y_all >= min_precip_threshold
    X = X_all[keep_mask]
    y = y_all[keep_mask]
    time_truncated = xr.DataArray(np.array(time_all)[keep_mask], dims=["time"])

    logging.info(f"Created lagged features with n_lags={n_lags}")
    logging.info(f"Lost {n_lags} days at beginning due to lagging")
    logging.info(f"Total valid days (after lagging): {n_valid_samples}")
    if min_precip_threshold > 0:
        n_filtered = np.sum(~keep_mask)
        logging.info(f"Min precip threshold: {min_precip_threshold} inches")
        logging.info(f"Days kept: {len(y)} ({len(y)/n_valid_samples*100:.1f}%)")
        logging.info(
            f"Days filtered out: {n_filtered} ({n_filtered/n_valid_samples*100:.1f}%)"
        )

    logging.info(f"Feature matrix shape: {X.shape} ({n_features} features)")
    logging.info(f"Target shape: {y.shape}")
    logging.debug(
        f"Feature names would be: temp_min_d0, temp_max_d0, "
        + ", ".join(
            [
                f"temp_min_d-{i}, temp_max_d-{i}, precip_d-{i}"
                for i in range(1, n_lags + 1)
            ]
        )
    )

    # Log precipitation statistics
    logging.info(f"Target precip: mean={np.mean(y):.3f}, std={np.std(y):.3f}")
    logging.info(f"Target precip range: min={np.min(y):.3f}, max={np.max(y):.3f}")

    return X, y, time_truncated


def compute_climatology_baseline(
    precip: np.ndarray, time_coord: xr.DataArray, min_precip_threshold: float = 0.0
) -> Dict[int, float]:
    """Compute climatological mean precipitation for each day-of-year.

    Args:
        precip: Precipitation values
        time_coord: Time coordinates
        min_precip_threshold: Only include days >= threshold in climatology

    Returns:
        Dictionary mapping day-of-year (1-366) to mean precipitation
    """
    doy = time_coord.dt.dayofyear.values

    # Calculate mean for each day of year
    climatology = {}
    for d in range(1, 367):  # Days 1-366
        mask = (doy == d) & (precip >= min_precip_threshold)
        if np.sum(mask) > 0:
            climatology[d] = np.mean(precip[mask])
        else:
            # Fallback: use overall mean for days meeting threshold
            overall_mask = precip >= min_precip_threshold
            if np.sum(overall_mask) > 0:
                climatology[d] = np.mean(precip[overall_mask])
            else:
                climatology[d] = 0.0

    logging.debug(f"Computed climatology for {len(climatology)} days of year")
    logging.debug(
        f"Climatology range: {min(climatology.values()):.3f} to {max(climatology.values()):.3f}"
    )

    return climatology


class PrecipDataset(Dataset):
    """PyTorch Dataset for precipitation regression."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def create_datasets(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, ...]]:
    """Split data and create DataLoader objects.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_arrays)
        where split_arrays = (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )

    logging.info(
        f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )
    logging.debug(
        f"Split proportions: Train={len(X_train)/len(X)*100:.1f}%, "
        f"Val={len(X_val)/len(X)*100:.1f}%, Test={len(X_test)/len(X)*100:.1f}%"
    )

    # Create datasets
    train_dataset = PrecipDataset(X_train, y_train)
    val_dataset = PrecipDataset(X_val, y_val)
    test_dataset = PrecipDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.debug(f"Created DataLoaders with batch_size={batch_size}")

    split_arrays = (X_train, X_val, X_test, y_train, y_val, y_test)
    return train_loader, val_loader, test_loader, split_arrays


class PrecipNet(nn.Module):
    """Simple feedforward network for precipitation regression.

    Architecture: input -> hidden layer -> output
    Single hidden layer with ReLU activation for pedagogical simplicity.
    """

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Single output (precipitation amount)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
) -> Tuple[nn.Module, Dict]:
    """Train the model using Adam optimizer and MSE loss.

    Returns:
        Tuple of (trained_model, training_history)
    """
    # MSE loss is appropriate for regression tasks
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logging.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
    logging.debug(f"Using criterion: {criterion.__class__.__name__}")
    logging.debug(f"Using optimizer: {optimizer.__class__.__name__}")

    # Track training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_features, batch_targets in train_loader:
            # Standard training loop for regression
            optimizer.zero_grad()  # Clear gradients
            outputs = model(batch_features)  # Forward pass
            loss = criterion(outputs, batch_targets)  # Calculate MSE
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        # Store in history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["epochs"].append(epoch + 1)

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1:3d}: Train Loss={epoch_train_loss:.6f}, "
                f"Val Loss={epoch_val_loss:.6f}"
            )

    logging.info("Training completed")
    return model, history


def evaluate_baselines(
    y_test: np.ndarray,
    time_test: xr.DataArray,
    climatology: Dict[int, float],
    precip_full: np.ndarray,
    time_full: xr.DataArray,
) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline models: climatology and persistence.

    Returns:
        Dictionary with metrics for each baseline model
    """
    logging.info("=== BASELINE EVALUATIONS ===")

    results = {}

    # Climatology baseline: predict day-of-year mean
    doy_test = time_test.dt.dayofyear.values
    clim_predictions = np.array([climatology[d] for d in doy_test])

    clim_mae = mean_absolute_error(y_test, clim_predictions)
    clim_rmse = np.sqrt(mean_squared_error(y_test, clim_predictions))
    clim_r2 = r2_score(y_test, clim_predictions)

    results["climatology"] = {"MAE": clim_mae, "RMSE": clim_rmse, "R2": clim_r2}

    logging.info("Climatology Baseline:")
    logging.info(f"  MAE:  {clim_mae:.4f} inches")
    logging.info(f"  RMSE: {clim_rmse:.4f} inches")
    logging.info(f"  R²:   {clim_r2:.4f}")

    # Persistence baseline: predict yesterday's precipitation
    # Need to find yesterday's precip for each test day
    # This is tricky with the train/test split, so we'll use the full dataset
    time_to_precip = dict(zip(time_full.values, precip_full))

    persist_predictions = []
    valid_indices = []

    for i, t in enumerate(time_test.values):
        # Find yesterday (1 day before)
        yesterday = t - np.timedelta64(1, "D")
        if yesterday in time_to_precip:
            persist_predictions.append(time_to_precip[yesterday])
            valid_indices.append(i)

    if len(persist_predictions) > 0:
        persist_predictions = np.array(persist_predictions)
        y_test_persist = y_test[valid_indices]

        persist_mae = mean_absolute_error(y_test_persist, persist_predictions)
        persist_rmse = np.sqrt(mean_squared_error(y_test_persist, persist_predictions))
        persist_r2 = r2_score(y_test_persist, persist_predictions)

        results["persistence"] = {
            "MAE": persist_mae,
            "RMSE": persist_rmse,
            "R2": persist_r2,
        }

        logging.info("Persistence Baseline:")
        logging.info(f"  MAE:  {persist_mae:.4f} inches")
        logging.info(f"  RMSE: {persist_rmse:.4f} inches")
        logging.info(f"  R²:   {persist_r2:.4f}")
        logging.debug(f"  (used {len(persist_predictions)}/{len(y_test)} test samples)")
    else:
        logging.warning("Could not compute persistence baseline")
        results["persistence"] = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate neural network model.

    Returns:
        Tuple of (metrics_dict, predictions_array)
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_features, _ in test_loader:
            outputs = model(batch_features)
            all_predictions.extend(outputs.cpu().numpy().flatten())

    predictions = np.array(all_predictions)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    logging.info("=== NEURAL NETWORK EVALUATION ===")
    logging.info(f"MAE:  {mae:.4f} inches")
    logging.info(f"RMSE: {rmse:.4f} inches")
    logging.info(f"R²:   {r2:.4f}")

    return metrics, predictions


def plot_training_curves(history: Dict) -> str:
    """Plot training and validation loss curves.

    Returns:
        Filename of saved plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        history["epochs"],
        history["train_loss"],
        "b-",
        label="Training Loss",
        linewidth=2,
    )
    plt.plot(
        history["epochs"],
        history["val_loss"],
        "r-",
        label="Validation Loss",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "outputs"
    os.makedirs(output_directory, exist_ok=True)
    filename = f"{output_directory}/precip_loss_curves_{timestamp}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    return filename


def plot_predictions(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> str:
    """Plot predicted vs actual precipitation with scatter and residuals.

    Returns:
        Filename of saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: predicted vs actual
    ax = axes[0]
    ax.scatter(y_test, predictions, alpha=0.3, s=10)
    max_val = max(y_test.max(), predictions.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Precipitation (inches)")
    ax.set_ylabel("Predicted Precipitation (inches)")
    ax.set_title("Predicted vs Actual Precipitation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual plot
    residuals = predictions - y_test
    ax = axes[1]
    ax.scatter(predictions, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Precipitation (inches)")
    ax.set_ylabel("Residual (Predicted - Actual)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "outputs"
    filename = f"{output_directory}/precip_predictions_{timestamp}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    return filename


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NYC Precipitation Regression Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--n-lags",
        type=int,
        default=7,
        help="Number of previous days to include as features",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Number of neurons in hidden layer",
    )

    parser.add_argument(
        "--min-precip-threshold",
        type=float,
        default=0.0,
        help="Minimum precipitation (inches) to include days (default 0.0 keeps all days, 0.01 filters to rainy days only)",
    )

    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()

    data_file = "data/central-park-station_daily-data_18690101-20230930.nc"

    # Set up logging
    setup_logging()

    logging.info("=== NYC PRECIPITATION REGRESSION ===")
    logging.info(
        f"Training parameters: epochs={args.epochs}, learning_rate={args.learning_rate}, "
        f"n_lags={args.n_lags}, hidden_size={args.hidden_size}, "
        f"min_precip_threshold={args.min_precip_threshold}"
    )

    # Load data
    logging.info("1. Loading data...")
    precip, temp_min, temp_max, time_coord = load_data(data_file)

    # Create lagged features
    logging.info("2. Creating lagged features...")
    X, y, time_truncated = create_lagged_features(
        precip,
        temp_min,
        temp_max,
        time_coord,
        n_lags=args.n_lags,
        min_precip_threshold=args.min_precip_threshold,
    )

    # Compute climatology for baseline
    logging.info("3. Computing climatology baseline...")
    # Remove NaNs from full dataset for climatology
    valid_mask = ~np.isnan(precip)
    climatology = compute_climatology_baseline(
        precip[valid_mask],
        time_coord[valid_mask],
        min_precip_threshold=args.min_precip_threshold,
    )

    # Create datasets
    logging.info("4. Creating train/val/test splits...")
    train_loader, val_loader, test_loader, split_arrays = create_datasets(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_arrays

    # We need time coordinates for test set to evaluate baselines
    # Split time_truncated the same way as X and y
    time_temp, time_test = train_test_split(
        time_truncated.values, test_size=0.2, random_state=42
    )
    # Convert back to datetime64
    time_test = xr.DataArray(time_test, dims=["time"])

    # Define model
    logging.info("5. Defining model...")
    n_features = X.shape[1]
    model = PrecipNet(n_features=n_features, hidden_size=args.hidden_size)
    logging.info(f"Model architecture: {model}")
    logging.debug(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    logging.info("6. Training model...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Plot training curves
    logging.info("7. Generating training curves...")
    loss_plot_file = plot_training_curves(history)
    logging.info(f"Loss curves saved as '{loss_plot_file}'")

    # Evaluate baselines
    logging.info("8. Evaluating baseline models...")
    baseline_results = evaluate_baselines(
        y_test,
        time_test,
        climatology,
        precip[valid_mask],
        time_coord[valid_mask],
    )

    # Evaluate neural network
    logging.info("9. Evaluating neural network...")
    nn_metrics, predictions = evaluate_model(model, test_loader, y_test)

    # Plot predictions
    logging.info("10. Generating prediction plots...")
    pred_plot_file = plot_predictions(y_test, predictions)
    logging.info(f"Prediction plots saved as '{pred_plot_file}'")

    # Compare all models
    logging.info("=== MODEL COMPARISON ===")
    logging.info(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    logging.info("-" * 56)

    for model_name, metrics in baseline_results.items():
        logging.info(
            f"{model_name.capitalize():<20} "
            f"{metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['R2']:<12.4f}"
        )

    logging.info(
        f"{'Neural Network':<20} "
        f"{nn_metrics['MAE']:<12.4f} {nn_metrics['RMSE']:<12.4f} {nn_metrics['R2']:<12.4f}"
    )

    logging.info("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
