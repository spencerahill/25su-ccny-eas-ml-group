"""
Train-Test Split Examples for Time Series Data
==============================================

This script demonstrates the correct and incorrect ways to split time series data
for machine learning, highlighting why temporal order matters.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def demonstrate_incorrect_splitting():
    """Show what happens when you use train_test_split incorrectly on time series."""
    
    print("=== INCORRECT: Random Splitting of Time Series ===\n")
    
    # Create a simple time series with trend
    time = np.arange(1000)
    trend = 0.1 * time  # Linear trend
    noise = np.random.normal(0, 1, 1000)
    data = trend + noise
    
    # INCORRECT: Using train_test_split with shuffle=True (default)
    X = time.reshape(-1, 1)
    y = data
    
    X_train_incorrect, X_test_incorrect, y_train_incorrect, y_test_incorrect = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print("Problem: Data is randomly shuffled, breaking temporal order!")
    print(f"Train indices range: {X_train_incorrect.min():.0f} to {X_train_incorrect.max():.0f}")
    print(f"Test indices range: {X_test_incorrect.min():.0f} to {X_test_incorrect.max():.0f}")
    print("This means future data might be in training set, past data in test set!")
    
    return X_train_incorrect, X_test_incorrect, y_train_incorrect, y_test_incorrect

def demonstrate_correct_splitting():
    """Show the correct way to split time series data."""
    
    print("\n=== CORRECT: Chronological Splitting of Time Series ===\n")
    
    # Create the same time series
    time = np.arange(1000)
    trend = 0.1 * time
    noise = np.random.normal(0, 1, 1000)
    data = trend + noise
    
    # CORRECT: Use train_test_split with shuffle=False
    X = time.reshape(-1, 1)
    y = data
    
    X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print("Solution: Set shuffle=False to maintain temporal order!")
    print(f"Train indices range: {X_train_correct.min():.0f} to {X_train_correct.max():.0f}")
    print(f"Test indices range: {X_test_correct.min():.0f} to {X_test_correct.max():.0f}")
    print("Now all training data comes before test data!")
    
    return X_train_correct, X_test_correct, y_train_correct, y_test_correct

def demonstrate_manual_splitting():
    """Show manual chronological splitting (most explicit for time series)."""
    
    print("\n=== MANUAL: Explicit Chronological Splitting ===\n")
    
    # Create the same time series
    time = np.arange(1000)
    trend = 0.1 * time
    noise = np.random.normal(0, 1, 1000)
    data = trend + noise
    
    # MANUAL: Explicit chronological split
    train_size = int(0.8 * len(time))
    
    X_train_manual = time[:train_size].reshape(-1, 1)
    y_train_manual = data[:train_size]
    X_test_manual = time[train_size:].reshape(-1, 1)
    y_test_manual = data[train_size:]
    
    print("Manual splitting gives you complete control over the split!")
    print(f"Train indices range: {X_train_manual.min():.0f} to {X_train_manual.max():.0f}")
    print(f"Test indices range: {X_test_manual.min():.0f} to {X_test_manual.max():.0f}")
    
    return X_train_manual, X_test_manual, y_train_manual, y_test_manual

def visualize_splits():
    """Visualize the different splitting approaches."""
    
    # Create data
    time = np.arange(1000)
    trend = 0.1 * time
    noise = np.random.normal(0, 1, 1000)
    data = trend + noise
    
    # Get different splits
    X_train_incorrect, X_test_incorrect, y_train_incorrect, y_test_incorrect = demonstrate_incorrect_splitting()
    X_train_correct, X_test_correct, y_train_correct, y_test_correct = demonstrate_correct_splitting()
    X_train_manual, X_test_manual, y_train_manual, y_test_manual = demonstrate_manual_splitting()
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot incorrect splitting
    axes[0].scatter(X_train_incorrect, y_train_incorrect, c='blue', alpha=0.6, s=10, label='Train')
    axes[0].scatter(X_test_incorrect, y_test_incorrect, c='red', alpha=0.6, s=10, label='Test')
    axes[0].set_title('INCORRECT: Random Shuffling (shuffle=True)')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot correct splitting
    axes[1].scatter(X_train_correct, y_train_correct, c='blue', alpha=0.6, s=10, label='Train')
    axes[1].scatter(X_test_correct, y_test_correct, c='red', alpha=0.6, s=10, label='Test')
    axes[1].set_title('CORRECT: Chronological Order (shuffle=False)')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot manual splitting
    axes[2].scatter(X_train_manual, y_train_manual, c='blue', alpha=0.6, s=10, label='Train')
    axes[2].scatter(X_test_manual, y_test_manual, c='red', alpha=0.6, s=10, label='Test')
    axes[2].set_title('MANUAL: Explicit Chronological Split')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def your_original_approach_fixed():
    """Show how to fix your original approach."""
    
    print("\n=== FIXING YOUR ORIGINAL APPROACH ===\n")
    
    # Your original data structure
    precip_data = np.random.random(1000)  # Simulated precipitation
    temp_data = np.random.random(1000)    # Simulated temperature
    
    print("Your original approach:")
    print("data_for_tts = np.array([ds_cp['precip'].values, ds_cp['temp_avg'].values])")
    print("tts = train_test_split(data_for_tts)  # This shuffles the data!")
    
    print("\nThe problem: train_test_split shuffles by default, breaking temporal order.")
    
    # Fixed approach
    print("\nFixed approach:")
    print("# 1. Create proper feature-target pairs")
    print("X = temp_data.reshape(-1, 1)  # Features (temperature)")
    print("y = precip_data               # Target (precipitation)")
    
    X = temp_data.reshape(-1, 1)
    y = precip_data
    
    print("\n# 2. Split without shuffling")
    print("X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    print(f"\nResult:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Run all demonstrations."""
    
    print("Time Series Train-Test Split Examples")
    print("=" * 50)
    
    # Run demonstrations
    your_original_approach_fixed()
    demonstrate_incorrect_splitting()
    demonstrate_correct_splitting()
    demonstrate_manual_splitting()
    
    # Show visualization
    visualize_splits()
    
    print("\n" + "=" * 50)
    print("KEY TAKEAWAYS:")
    print("1. For time series, NEVER use shuffle=True (default)")
    print("2. Use shuffle=False with train_test_split")
    print("3. Or manually split chronologically")
    print("4. Always maintain temporal order: past → train, future → test")

if __name__ == "__main__":
    main() 