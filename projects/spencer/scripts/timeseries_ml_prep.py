"""
Time Series ML Preparation Script
================================

This script demonstrates how to properly prepare time series data for machine learning,
including proper train/validation/test splits that maintain temporal order.
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load the Central Park weather data and prepare it for ML modeling."""
    
    # Load the data
    path_cp_data = "../data/central-park-station_daily-data_18690101-20230930.nc"
    ds_cp = xr.open_dataset(path_cp_data)
    
    # Extract the time series data
    precip_data = ds_cp["precip"].values
    temp_data = ds_cp["temp_avg"].values
    time_index = ds_cp["time"].values
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'time': time_index,
        'precip': precip_data,
        'temp_avg': temp_data
    })
    
    # Remove any NaN values
    df = df.dropna()
    
    print(f"Data shape after removing NaN: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df

def split_time_series_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split time series data chronologically into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    train_ratio : float
        Proportion of data for training (default: 0.70)
    val_ratio : float
        Proportion of data for validation (default: 0.15)
    test_ratio : float
        Proportion of data for testing (default: 0.15)
    
    Returns:
    --------
    tuple : (train_df, val_df, test_df)
    """
    
    # Calculate split indices
    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    
    # Split chronologically (maintain temporal order)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def prepare_features_and_targets(train_df, val_df, test_df, 
                                predict_precip_from_temp=True):
    """
    Prepare features and targets for ML modeling.
    
    Parameters:
    -----------
    train_df, val_df, test_df : pandas.DataFrame
        Split datasets
    predict_precip_from_temp : bool
        If True, predict precipitation from temperature
        If False, predict temperature from precipitation
    
    Returns:
    --------
    tuple : (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    
    if predict_precip_from_temp:
        # Predict precipitation from temperature
        X_train = train_df[['temp_avg']].values
        y_train = train_df['precip'].values
        X_val = val_df[['temp_avg']].values
        y_val = val_df['precip'].values
        X_test = test_df[['temp_avg']].values
        y_test = test_df['precip'].values
        feature_name = 'temperature'
        target_name = 'precipitation'
    else:
        # Predict temperature from precipitation
        X_train = train_df[['precip']].values
        y_train = train_df['temp_avg'].values
        X_val = val_df[['precip']].values
        y_val = val_df['temp_avg'].values
        X_test = test_df[['precip']].values
        y_test = test_df['temp_avg'].values
        feature_name = 'precipitation'
        target_name = 'temperature'
    
    print(f"\nPredicting {target_name} from {feature_name}")
    print(f"Feature shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Scale the features (important for many ML models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nScaled feature ranges:")
    print(f"X_train_scaled: {X_train_scaled.min():.2f} to {X_train_scaled.max():.2f}")
    print(f"X_val_scaled: {X_val_scaled.min():.2f} to {X_val_scaled.max():.2f}")
    print(f"X_test_scaled: {X_test_scaled.min():.2f} to {X_test_scaled.max():.2f}")
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, scaler, feature_name, target_name)

def visualize_splits(df, train_df, val_df, test_df):
    """Visualize the temporal splits of the data."""
    
    plt.figure(figsize=(15, 8))
    
    # Plot temperature
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['temp_avg'], 'b-', alpha=0.3, label='All data')
    plt.plot(train_df['time'], train_df['temp_avg'], 'b-', label='Train', linewidth=2)
    plt.plot(val_df['time'], val_df['temp_avg'], 'orange', label='Validation', linewidth=2)
    plt.plot(test_df['time'], test_df['temp_avg'], 'red', label='Test', linewidth=2)
    plt.title('Temperature Data Split')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precipitation
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['precip'], 'g-', alpha=0.3, label='All data')
    plt.plot(train_df['time'], train_df['precip'], 'g-', label='Train', linewidth=2)
    plt.plot(val_df['time'], val_df['precip'], 'orange', label='Validation', linewidth=2)
    plt.plot(test_df['time'], test_df['precip'], 'red', label='Test', linewidth=2)
    plt.title('Precipitation Data Split')
    plt.ylabel('Precipitation (inches)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate the complete workflow."""
    
    print("=== Time Series ML Data Preparation ===\n")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Split the data
    train_df, val_df, test_df = split_time_series_data(df)
    
    # Visualize the splits
    visualize_splits(df, train_df, val_df, test_df)
    
    # Prepare features and targets (predict precipitation from temperature)
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, feature_name, target_name) = prepare_features_and_targets(
        train_df, val_df, test_df, predict_precip_from_temp=True
    )
    
    print(f"\n=== Data Ready for ML Modeling ===")
    print(f"Your data is now ready for training ML models!")
    print(f"Use X_train, y_train for training")
    print(f"Use X_val, y_val for validation during training")
    print(f"Use X_test, y_test for final evaluation")
    print(f"Use scaler to transform new data")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler)

if __name__ == "__main__":
    # Run the complete workflow
    data_splits = main() 