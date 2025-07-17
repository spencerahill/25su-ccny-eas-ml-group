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
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_explore_data(filepath: str) -> tuple[np.ndarray, xr.DataArray]:
    """Load temperature data from netCDF file."""
    ds = xr.open_dataset(filepath)
    
    # Extract temperature and time coordinate
    temp_avg = ds["temp_avg"].values
    time_coord = ds["time"]
    
    # Close dataset
    ds.close()
    
    print(f"Loaded {len(temp_avg)} temperature records")
    print(f"Date range: {time_coord[0].dt.strftime('%Y-%m-%d').values} to {time_coord[-1].dt.strftime('%Y-%m-%d').values}")
    print(f"Temperature range: {np.nanmin(temp_avg):.1f}°F to {np.nanmax(temp_avg):.1f}°F")
    
    return temp_avg, time_coord


def preprocess_data(temperatures: np.ndarray, time_coord: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear detrending and normalization, create seasonal labels."""
    # Remove any NaN values
    valid_mask = ~np.isnan(temperatures)
    clean_temps = temperatures[valid_mask]
    clean_time = time_coord[valid_mask]
    
    print(f"Removed {np.sum(~valid_mask)} NaN values")
    
    # Linear detrending
    years = clean_time.dt.year.values
    years_numeric = years - years.min()  # Start from 0 for numerical stability
    
    # Fit linear trend
    reg = LinearRegression()
    reg.fit(years_numeric.reshape(-1, 1), clean_temps)
    trend = reg.predict(years_numeric.reshape(-1, 1))
    
    # Remove trend but keep mean temperature level
    detrended_temps = clean_temps - trend + np.mean(clean_temps)
    
    print(f"Linear trend: {reg.coef_[0]:.3f}°F per year")
    
    # Z-score normalization
    temp_mean = np.mean(detrended_temps)
    temp_std = np.std(detrended_temps)
    normalized_temps = (detrended_temps - temp_mean) / temp_std
    
    print(f"Normalized temperature: mean={temp_mean:.1f}°F, std={temp_std:.1f}°F")
    
    # Create seasonal labels
    months = clean_time.dt.month.values
    labels = np.zeros(len(months), dtype=int)
    
    # Winter: Dec, Jan, Feb (0)
    labels[(months == 12) | (months == 1) | (months == 2)] = 0
    
    # Summer: Jun, Jul, Aug (1)
    labels[(months == 6) | (months == 7) | (months == 8)] = 1
    
    # Spring/Fall: Mar, Apr, May, Sep, Oct, Nov (2)
    labels[(months == 3) | (months == 4) | (months == 5) | 
           (months == 9) | (months == 10) | (months == 11)] = 2
    
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ["Winter", "Summer", "Spring/Fall"]
    for i, (label, count) in enumerate(zip(unique, counts)):
        pct = count / len(labels) * 100
        print(f"{class_names[label]}: {count} days ({pct:.1f}%)")
    
    return normalized_temps, labels


class TempSeasonDataset(Dataset):
    """PyTorch Dataset for temperature to season classification."""
    
    def __init__(self, temperatures: np.ndarray, labels: np.ndarray):
        self.temperatures = torch.tensor(temperatures, dtype=torch.float32).reshape(-1, 1)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.temperatures[idx], self.labels[idx]


def create_datasets(X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.6, val_ratio: float = 0.2, 
                   batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
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
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create datasets
    train_dataset = TempSeasonDataset(X_train, y_train)
    val_dataset = TempSeasonDataset(X_val, y_val)
    test_dataset = TempSeasonDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class TempSeasonNet(nn.Module):
    """Simple feedforward network for temperature to season classification."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                epochs: int = 100, learning_rate: float = 0.01) -> nn.Module:
    """Train the model using SGD optimizer."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_temps, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_temps)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_temps, batch_labels in val_loader:
                outputs = model(batch_temps)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.1f}%, Val Loss={val_loss/len(val_loader):.4f}, "
                  f"Val Acc={val_acc:.1f}%")
    
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> None:
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
    
    print("\\n=== MODEL EVALUATION ===")
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\\nPer-class Metrics:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Confusion matrix
    print("\\nConfusion Matrix:")
    print("Predicted ->")
    print(f"{'Actual':<12} {'Winter':<8} {'Summer':<8} {'Spring/Fall':<8}")
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:<12}"
        for j in range(3):
            row += f"{cm[i,j]:<8}"
        print(row)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()


def main():
    """Main execution pipeline."""
    # File path
    data_file = "data/central-park-station_daily-data_18690101-20230930.nc"
    
    print("=== NYC TEMPERATURE TO SEASON CLASSIFICATION ===\\n")
    
    # Load data
    print("1. Loading data...")
    temperatures, time_coord = load_and_explore_data(data_file)
    
    # Preprocess data
    print("\\n2. Preprocessing data...")
    X, y = preprocess_data(temperatures, time_coord)
    
    # Create datasets
    print("\\n3. Creating datasets...")
    train_loader, val_loader, test_loader = create_datasets(X, y)
    
    # Define model
    print("\\n4. Defining model...")
    model = TempSeasonNet()
    print(f"Model architecture: {model}")
    
    # Train model
    print("\\n5. Training model...")
    model = train_model(model, train_loader, val_loader)
    
    # Evaluate model
    print("\\n6. Evaluating model...")
    evaluate_model(model, test_loader)
    
    print("\\n=== COMPLETE ===")


if __name__ == "__main__":
    main()