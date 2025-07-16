"""
Add Validation Split to Your Existing Code
=========================================

This script shows how to modify your existing train_test_split code
to include a proper validation set.
"""

# Your existing code (from your notebook):
# Extract your data properly
precip_data = ds_cp["precip"].values
temp_data = ds_cp["temp_avg"].values

# Create feature-target pairs
X = temp_data.reshape(-1, 1)  # Features (temperature)
y = precip_data  # Target (precipitation)

# Split chronologically (no shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

print("=== Your Current Split ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ADD THIS CODE TO CREATE A THREE-WAY SPLIT:

print("\n=== Creating Three-Way Split (Train/Validation/Test) ===")

# Method 1: Manual chronological split (recommended for time series)
train_size = int(0.75 * len(X))  # 75% for training
val_size = int(0.15 * len(X))    # 15% for validation
# Remaining 10% for testing

# Split chronologically
X_train_final = X[:train_size]
y_train_final = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test_final = X[train_size + val_size:]
y_test_final = y[train_size + val_size:]

print(f"Original data size: {len(X)}")
print(f"Training set: {len(X_train_final)} samples ({len(X_train_final)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test_final)} samples ({len(X_test_final)/len(X)*100:.1f}%)")

# Method 2: Using train_test_split twice (alternative approach)
print("\n=== Alternative Method: Using train_test_split twice ===")

# First split: 85% train+val, 15% test
X_temp, X_test_split2, y_temp, y_test_split2 = train_test_split(
    X, y, shuffle=False, test_size=0.15
)

# Second split: 75% train, 15% val (of original data)
# This means val is 15%/85% = 17.6% of the temp data
X_train_split2, X_val_split2, y_train_split2, y_val_split2 = train_test_split(
    X_temp, y_temp, shuffle=False, test_size=0.176  # 0.15/0.85 ≈ 0.176
)

print(f"Training set: {len(X_train_split2)} samples ({len(X_train_split2)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val_split2)} samples ({len(X_val_split2)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test_split2)} samples ({len(X_test_split2)/len(X)*100:.1f}%)")

print("\n=== Recommendation ===")
print("Use Method 1 (manual split) for time series data.")
print("It's simpler and gives you exact control over the split ratios.")
print("Your final variables will be:")
print("- X_train_final, y_train_final (for training)")
print("- X_val, y_val (for validation during training)")
print("- X_test_final, y_test_final (for final evaluation)") 