# NYC Temperature → Season Classification Model Design

## Problem Statement
**Input**: Daily average temperature timeseries (NYC)
**Output**: Seasonal classification (3 classes: Winter, Summer, Spring/Fall)
**Goal**: Predict season from temperature using simplest possible feedforward neural network

## Design Decisions

### 1. Model Architecture
**DECIDED**: Single hidden layer feedforward network
- Input: 1 neuron (temperature)
- Hidden: 32 neurons with ReLU activation
- Output: 3 neurons (Winter, Summer, Spring/Fall)
- No activation on output layer (CrossEntropyLoss handles softmax)

### 2. Class Definitions
**DECIDED**: 3-class seasonal grouping
- Winter: Dec, Jan, Feb
- Summer: Jun, Jul, Aug  
- Spring/Fall: Mar, Apr, May, Sep, Oct, Nov

### 3. Loss Function
**DECIDED**: CrossEntropyLoss
- Appropriate for mutually exclusive multiclass classification
- Handles softmax conversion internally
- Standard choice for this problem type

### 4. Data Preprocessing
**DECIDED**: Simple linear detrending + z-score normalization
- Remove secular warming trend: fit linear regression temp ~ year, subtract trend
- Z-score normalize detrended temperatures: (temp - mean) / std
- Keep extreme weather events (no outlier removal)
- Use all available years of data

### 5. Data Splitting Strategy
**DECIDED**: Random shuffle split
- Standard train/validation/test proportions
- Random rather than chronological split since relationship is static
- Detrending handles temporal drift

### 6. Training Configuration
**DECIDED**: Basic setup
- Optimizer: SGD (user preference for familiarity)
- Learning rate: Constant rate (no scheduling)
- Batch size: Constant size appropriate for ~56k daily data points
- Epochs: Constant number

### 7. Evaluation Metrics
**DECIDED**: Comprehensive evaluation with 4 metrics
- Primary: Overall accuracy
- Secondary: Confusion matrix (visual understanding of errors)
- Tertiary: Per-class precision/recall (important given class imbalance)
- Quaternary: Per-class F1 scores (balanced performance measure)

**Class distribution**: Winter (24.7%), Summer (25.2%), Spring/Fall (50.1%) - imbalanced
**Class balancing**: Use unweighted loss initially, add weighted loss if results show systematic bias

### 8. Model Regularization
**DECIDED**: Start with no regularization, add if needed
- **Initial approach**: No regularization (simple model, large dataset, clear patterns)
- **Future options if overfitting occurs**:
  - Dropout: Add `nn.Dropout(0.2)` after hidden layer
  - Weight decay: Add `weight_decay=1e-4` to SGD optimizer  
  - Early stopping: Stop training when validation loss stops improving

### 9. Hyperparameter Specifics
**DECIDED**: Starting values
- Learning rate: 0.01 (good SGD starting point, reduce to 0.001 if unstable)
- Batch size: 64 (reasonable for ~56k dataset and CPU training)
- Number of epochs: 100 (simple problem should converge quickly)
- Train/validation/test split: 60/20/20 (standard proportions)