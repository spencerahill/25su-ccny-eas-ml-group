"""
NYC Temperature to Season Classification Model

Predicts season (Winter, Summer, Spring/Fall) from daily average temperature.
Uses simple feedforward neural network with linear detrending and normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import argparse
from datetime import datetime


def setup_logging() -> None:
    """Set up logging with both console and file handlers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_filename = f"{log_directory}/temp_season_classifier_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Console output
        ],
    )

    # Set console handler to INFO level, file handler to DEBUG
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

    # Get root logger and configure
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Suppress noisy matplotlib debug messages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_filename}")


def load_and_explore_data(filepath: str) -> tuple[np.ndarray, xr.DataArray]:
    """Load temperature data from netCDF file."""
    with xr.open_dataset(filepath) as ds:
        # Extract temperature and time coordinate
        temp_avg = ds["temp_avg"].values
        time_coord = ds["time"]

    logging.info(f"Loaded {len(temp_avg)} temperature records")
    logging.info(
        f"Date range: {time_coord[0].dt.strftime('%Y-%m-%d').values} to {time_coord[-1].dt.strftime('%Y-%m-%d').values}"
    )
    logging.info(
        f"Temperature range: {np.nanmin(temp_avg):.1f}°F to {np.nanmax(temp_avg):.1f}°F"
    )
    logging.debug(f"Temperature data shape: {temp_avg.shape}")

    return temp_avg, time_coord


def preprocess_data(
    temperatures: np.ndarray, time_coord: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear detrending and normalization, create seasonal labels."""
    # Remove any NaN values
    valid_mask = ~np.isnan(temperatures)
    clean_temps = temperatures[valid_mask]
    clean_time = time_coord[valid_mask]

    logging.info(f"Removed {np.sum(~valid_mask)} NaN values")
    logging.debug(f"Clean data shape: {clean_temps.shape}")

    # Linear detrending
    years = clean_time.dt.year.values
    years_numeric = years - years.min()  # Start from 0 for numerical stability

    # Fit linear trend
    reg = LinearRegression()
    reg.fit(years_numeric.reshape(-1, 1), clean_temps)
    trend = reg.predict(years_numeric.reshape(-1, 1))

    # Remove trend but keep mean temperature level
    detrended_temps = clean_temps - trend + np.mean(clean_temps)

    logging.info(f"Linear trend: {reg.coef_[0]:.3f}°F per year")
    logging.debug(
        f"Trend R² score: {reg.score(years_numeric.reshape(-1, 1), clean_temps):.3f}"
    )

    # Z-score normalization
    temp_mean = np.mean(detrended_temps)
    temp_std = np.std(detrended_temps)
    normalized_temps = (detrended_temps - temp_mean) / temp_std

    logging.info(
        f"Normalized temperature: mean={temp_mean:.1f}°F, std={temp_std:.1f}°F"
    )
    logging.debug(
        f"Normalized temperature range: {np.min(normalized_temps):.2f} to {np.max(normalized_temps):.2f}"
    )

    # Create seasonal labels
    months = clean_time.dt.month.values
    labels = np.zeros(len(months), dtype=int)

    # Winter: Dec, Jan, Feb (0)
    labels[(months == 12) | (months == 1) | (months == 2)] = 0

    # Summer: Jun, Jul, Aug (1)
    labels[(months == 6) | (months == 7) | (months == 8)] = 1

    # Spring/Fall: Mar, Apr, May, Sep, Oct, Nov (2)
    labels[
        (months == 3)
        | (months == 4)
        | (months == 5)
        | (months == 9)
        | (months == 10)
        | (months == 11)
    ] = 2

    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Winter", "Summer", "Spring/Fall"]
    for i, (label, count) in enumerate(zip(unique, counts)):
        pct = count / len(labels) * 100
        logging.info(f"{class_names[label]}: {count} days ({pct:.1f}%)")

    logging.debug(f"Label distribution: {dict(zip(unique, counts))}")

    return normalized_temps, labels


class TempSeasonDataset(Dataset):
    """PyTorch Dataset for temperature to season classification."""

    def __init__(self, temperatures: np.ndarray, labels: np.ndarray):
        self.temperatures = torch.tensor(temperatures, dtype=torch.float32).reshape(
            -1, 1
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.temperatures[idx], self.labels[idx]


def create_datasets(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split data and create DataLoader objects."""
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    logging.info(
        f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    )
    logging.debug(
        f"Split proportions: Train={len(X_train)/len(X)*100:.1f}%, Val={len(X_val)/len(X)*100:.1f}%, Test={len(X_test)/len(X)*100:.1f}%"
    )

    # Create datasets
    train_dataset = TempSeasonDataset(X_train, y_train)
    val_dataset = TempSeasonDataset(X_val, y_val)
    test_dataset = TempSeasonDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.debug(f"Created DataLoaders with batch_size={batch_size}")

    return train_loader, val_loader, test_loader


class TempSeasonNet(nn.Module):
    """Simple feedforward network for temperature to season classification."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def calculate_epoch_metrics(
    train_loss: float,
    val_loss: float,
    train_correct: int,
    val_correct: int,
    train_total: int,
    val_total: int,
    train_loader_len: int,
    val_loader_len: int,
) -> tuple[float, float, float, float]:
    """Calculate epoch-level metrics from batch accumulations."""
    epoch_train_loss = train_loss / train_loader_len
    epoch_val_loss = val_loss / val_loader_len
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    return epoch_train_loss, epoch_val_loss, train_acc, val_acc


def store_metrics_in_history(
    history: dict,
    epoch_train_loss: float,
    epoch_val_loss: float,
    train_acc: float,
    val_acc: float,
    epoch: int,
) -> None:
    """Store epoch metrics in training history dictionary."""
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["epochs"].append(epoch + 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> tuple[nn.Module, dict]:
    """Train the model using SGD optimizer.

    Returns:
        tuple: (trained_model, training_history)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    logging.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
    logging.debug(f"Using criterion: {criterion.__class__.__name__}")
    logging.debug(f"Using optimizer: {optimizer.__class__.__name__}")

    # Track training history for overfitting diagnostics
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs": [],
    }

    for epoch in range(epochs):
        # Training phase - model learns from data
        model.train()  # Enable training mode (affects dropout, batch norm)
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (
            batch_temps,
            batch_labels,
        ) in train_loader:  # Process in batches (typically 64 samples)
            # Core training steps for each batch:
            optimizer.zero_grad()  # 1. Clear gradients from previous batch
            outputs = model(batch_temps)  # 2. Forward pass: get predictions
            loss = criterion(outputs, batch_labels)  # 3. Calculate how wrong we are
            loss.backward()  # 4. Backward pass: compute gradients
            optimizer.step()  # 5. Update weights using gradients

            # Track statistics for monitoring
            train_loss += loss.item()  # Accumulate loss across batches
            _, predicted = torch.max(
                outputs.data, 1
            )  # Get predicted class (0, 1, or 2)
            train_total += batch_labels.size(0)  # Count total samples
            train_correct += (
                (predicted == batch_labels).sum().item()
            )  # Count correct predictions

        # Validation phase - test current model without learning
        model.eval()  # Switch to evaluation mode (disables dropout, etc.)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation (saves memory, speeds up)
            for batch_temps, batch_labels in val_loader:
                # Only forward pass - no learning, just measuring performance
                outputs = model(batch_temps)  # Get predictions
                loss = criterion(outputs, batch_labels)  # Calculate loss

                # Track validation statistics (same as training, but no weight updates)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate epoch metrics
        epoch_train_loss, epoch_val_loss, train_acc, val_acc = calculate_epoch_metrics(
            train_loss,
            val_loss,
            train_correct,
            val_correct,
            train_total,
            val_total,
            len(train_loader),
            len(val_loader),
        )

        # Store metrics in history
        store_metrics_in_history(
            history, epoch_train_loss, epoch_val_loss, train_acc, val_acc, epoch
        )

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1:3d}: Train Loss={epoch_train_loss:.4f}, "
                f"Train Acc={train_acc:.1f}%, Val Loss={epoch_val_loss:.4f}, "
                f"Val Acc={val_acc:.1f}%"
            )

    logging.info("Training completed")
    return model, history


def plot_loss_curves(history: dict) -> str:
    """Plot training and validation loss curves.

    Returns:
        str: Filename of saved plot
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
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "outputs"
    os.makedirs(output_directory, exist_ok=True)
    loss_filename = f"{output_directory}/loss_curves_{timestamp}.png"

    plt.savefig(loss_filename, dpi=150, bbox_inches="tight")
    plt.close()

    return loss_filename


def plot_accuracy_curves(history: dict) -> str:
    """Plot training and validation accuracy curves.

    Returns:
        str: Filename of saved plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        history["epochs"],
        history["train_acc"],
        "b-",
        label="Training Accuracy",
        linewidth=2,
    )
    plt.plot(
        history["epochs"],
        history["val_acc"],
        "r-",
        label="Validation Accuracy",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "outputs"
    os.makedirs(output_directory, exist_ok=True)
    acc_filename = f"{output_directory}/accuracy_curves_{timestamp}.png"

    plt.savefig(acc_filename, dpi=150, bbox_inches="tight")
    plt.close()

    return acc_filename


def evaluate_temperature_baseline(
    temperatures: np.ndarray, labels: np.ndarray
) -> float:
    """Evaluate simple temperature percentile-based baseline."""
    temp_25th = np.percentile(temperatures, 25)
    temp_75th = np.percentile(temperatures, 75)

    logging.info("=== TEMPERATURE BASELINE EVALUATION ===")
    logging.info(
        f"Temperature thresholds: 25th percentile = {temp_25th:.1f}°F, 75th percentile = {temp_75th:.1f}°F"
    )

    # Create predictions: coldest 25% → Winter, hottest 25% → Summer, middle 50% → Spring/Fall
    predictions = np.full_like(labels, 2)  # Default to Spring/Fall
    predictions[temperatures < temp_25th] = 0  # Winter
    predictions[temperatures > temp_75th] = 1  # Summer

    # Calculate metrics
    baseline_accuracy = accuracy_score(labels, predictions)
    baseline_cm = confusion_matrix(labels, predictions)

    class_names = ["Winter", "Summer", "Spring/Fall"]

    logging.info(
        f"Baseline Accuracy: {baseline_accuracy:.3f} ({baseline_accuracy*100:.1f}%)"
    )

    # Classification report
    logging.info("Baseline Per-class Metrics:")
    baseline_report = classification_report(
        labels, predictions, target_names=class_names
    )
    logging.info(f"Baseline Classification Report:\n{baseline_report}")

    # Confusion matrix
    logging.info("Baseline Confusion Matrix:")
    logging.info("Predicted ->")
    logging.info(f"{'Actual':<12} {'Winter':<8} {'Summer':<8} {'Spring/Fall':<8}")
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:<12}"
        for j in range(3):
            row += f"{baseline_cm[i,j]:<8}"
        logging.info(row)

    return baseline_accuracy


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model performance with comprehensive metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_temps, batch_labels in test_loader:
            outputs = model(batch_temps)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    class_names = ["Winter", "Summer", "Spring/Fall"]

    logging.info("=== MODEL EVALUATION ===")
    logging.info(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Classification report
    logging.info("Per-class Metrics:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    logging.info(f"Classification Report:\n{report}")

    # Confusion matrix
    logging.info("Confusion Matrix:")
    logging.info("Predicted ->")
    logging.info(f"{'Actual':<12} {'Winter':<8} {'Summer':<8} {'Spring/Fall':<8}")
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:<12}"
        for j in range(3):
            row += f"{cm[i,j]:<8}"
        logging.info(row)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    # Create outputs directory and timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "outputs"
    os.makedirs(output_directory, exist_ok=True)
    confusion_matrix_filename = f"{output_directory}/confusion_matrix_{timestamp}.png"

    plt.savefig(confusion_matrix_filename, dpi=150, bbox_inches="tight")
    logging.info(f"Confusion matrix saved as '{confusion_matrix_filename}'")
    plt.close()

    return accuracy


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NYC Temperature to Season Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for SGD optimizer",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    return parser.parse_args()


def main():
    """Main execution pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # File path
    data_file = "data/central-park-station_daily-data_18690101-20230930.nc"

    # Set up logging first
    setup_logging()

    logging.info("=== NYC TEMPERATURE TO SEASON CLASSIFICATION ===")
    logging.info(
        f"Training parameters: epochs={args.epochs}, learning_rate={args.learning_rate}"
    )

    # Load data
    logging.info("1. Loading data...")
    temperatures, time_coord = load_and_explore_data(data_file)

    # Preprocess data
    logging.info("2. Preprocessing data...")
    X, y = preprocess_data(temperatures, time_coord)

    # Evaluate temperature baseline before train/test split
    logging.info("3. Evaluating temperature baseline...")
    clean_temps = temperatures[~np.isnan(temperatures)]  # Remove NaN values to match y
    baseline_accuracy = evaluate_temperature_baseline(clean_temps, y)

    # Create datasets
    logging.info("4. Creating datasets...")
    train_loader, val_loader, test_loader = create_datasets(X, y)

    # Define model
    logging.info("5. Defining model...")
    model = TempSeasonNet()
    logging.info(f"Model architecture: {model}")

    # Train model
    logging.info("6. Training model...")
    model, training_history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Plot training curves for overfitting analysis
    logging.info("6b. Generating training curves...")
    loss_plot_file = plot_loss_curves(training_history)
    acc_plot_file = plot_accuracy_curves(training_history)
    logging.info(f"Loss curves saved as '{loss_plot_file}'")
    logging.info(f"Accuracy curves saved as '{acc_plot_file}'")

    # Evaluate ML model
    logging.info("7. Evaluating ML model...")
    ml_accuracy = evaluate_model(model, test_loader)

    # Compare performance
    logging.info("=== PERFORMANCE COMPARISON ===")
    logging.info(f"Temperature Baseline: {baseline_accuracy*100:.1f}% accuracy")
    logging.info(f"ML Model: {ml_accuracy*100:.1f}% accuracy")
    improvement = ml_accuracy - baseline_accuracy
    logging.info(
        f"ML Improvement: +{improvement*100:.1f} percentage points ({improvement/baseline_accuracy*100:.1f}% relative improvement)"
    )

    logging.info("=== COMPLETE ===")


if __name__ == "__main__":
    main()
