# ML Development Log: NYC Temperature to Season Classification

*A pedagogical record of the machine learning development process*

## Overview

This log documents the iterative development of a temperature-to-season classification model, capturing the real decision-making process, surprises, and lessons learned that students should understand about ML development.

**Problem**: Classify NYC daily temperatures into seasons (Winter, Summer, Spring/Fall)
**Data**: Central Park weather station data (1869-2023)
**Goal**: Educational - demonstrate ML fundamentals and proper methodology

---

## 2025-07-25: Initial Model Implementation

**Session Goals**: 
- Implement basic feedforward neural network
- Set up proper train/validation/test splits
- Create evaluation pipeline

**Approach**:
- Simple 2-layer NN (1 → 32 → 3 neurons)
- Linear detrending to remove climate trend
- Z-score normalization
- SGD optimizer, CrossEntropy loss

**Results**:
- Model accuracy: ~76%
- Training appeared stable
- No obvious overfitting with default parameters

**Key Decisions**:
- Used calendar months for ground truth labels (Dec/Jan/Feb = Winter, etc.)
- 60/20/20 train/val/test split with stratification
- Minimal architecture to start simple

**Questions Raised**:
- Is 76% accuracy actually good for this problem?
- What would a non-ML baseline achieve?

---

## 2025-07-25: Overfitting Diagnostics Implementation

**Hypothesis**: Need diagnostic tools to understand training behavior and detect overfitting

**Implementation**:
- Added training history tracking (loss/accuracy over epochs)
- Created separate plotting functions for loss and accuracy curves
- Modified `train_model()` to return both model and training history
- Added command-line arguments for easy parameter experimentation

**Overfitting Experiment**:
- Increased learning rate from 0.01 → 0.1
- Extended training to 200 epochs
- **Result**: Successfully induced overfitting patterns!

**Key Observations**:
- Higher learning rate created volatile validation metrics
- Training loss continued decreasing while validation became erratic
- Perfect teaching example of overfitting curves

**Pedagogical Value**:
- Students can now see healthy vs. unhealthy training patterns
- Visual diagnostics more intuitive than just final accuracy numbers
- Command-line interface allows easy experimentation

**Technical Lessons**:
- Encapsulated functions better than monolithic training loop
- Proper separation of concerns (metrics calculation, plotting, storage)
- Always inspect plots, don't just check if files exist

---

## 2025-07-25: The Baseline Reality Check

**Hypothesis**: Our neural network should significantly outperform simple rule-based approaches

**Motivation**: 
- Student question: "How good is 75-76% accuracy really?"
- Class distribution: 50% Spring/Fall, 25% each Winter/Summer
- Always-predict-Spring/Fall baseline = 50% accuracy

**Elegant Baseline Implementation**:
- Used temperature percentiles: < 25th percentile = Winter, > 75th percentile = Summer
- Evaluated on original temperature data before ML preprocessing
- Clean implementation without data conversion complexity

**Shocking Results**:
- **Temperature Baseline: 75.6% accuracy**
- **ML Model: 75.7% accuracy**  
- **ML Improvement: +0.1 percentage points (0.1% relative improvement)**

**Baseline Thresholds**:
- Winter: < 40.0°F (25th percentile)
- Summer: > 69.5°F (75th percentile)
- Spring/Fall: 40.0°F - 69.5°F (middle 50%)

**Profound Lessons**:
1. **ML isn't magic** - sometimes simple domain knowledge works just as well
2. **Context is everything** - raw accuracy numbers are meaningless without baselines
3. **Always establish baselines first** - should be step 1 in any ML project
4. **Simple can be powerful** - temperature percentiles capture the underlying pattern

**Pedagogical Impact**:
- Humbling reality check for students who think ML always wins
- Demonstrates proper experimental methodology
- Shows when NOT to use ML (or when simpler approaches suffice)
- Reinforces importance of domain knowledge

**Questions for Next Session**:
- Why isn't ML helping more? Is the problem too simple?
- Should we try more complex architectures?
- What other baselines could we test?
- How do we know when ML is the right tool?

---

## Development Insights Running List

### Key ML Lessons Learned:
1. **Always implement meaningful baselines before celebrating ML results**
2. **Visual diagnostics (training curves) are more valuable than final metrics alone**
3. **Overfitting can be induced with higher learning rates for teaching purposes**
4. **Domain knowledge baselines can be surprisingly competitive**
5. **Command-line interfaces make experimentation much easier**

### Technical Best Practices Discovered:
1. **Encapsulate logic into focused functions** (calculate_epoch_metrics, store_metrics_in_history)
2. **Always inspect generated plots**, don't just check file existence
3. **Evaluate baselines on original data** before ML preprocessing
4. **Use proper logging with timestamps** for experiment tracking
5. **Git commit after every nontrivial change** (learned this the hard way!)

### Common Pitfalls Avoided:
1. **Hardcoding normalization parameters** instead of passing them through pipeline
2. **Forgetting to commit changes** during development
3. **Over-engineering** the baseline comparison with unnecessary data conversions
4. **Assuming ML will always beat simple approaches**

---

---

## 2025-07-25 (Evening): Regression Problem Design Decision

**Context**: After successful classification implementation, need to add regression component for complete pedagogical coverage.

**Problem Selection Process**:
Considered multiple regression targets:
- Daily temperature range (max - min)
- Growing degree days (agricultural relevance)
- Precipitation amount (challenging due to skewness)
- **Next-day temperature prediction** ← SELECTED

**Decision Rationale**:
1. **Immediate student relevance**: Weather forecasting is universally understood
2. **Clean pedagogical contrast**: Same dataset, different problem type (classification vs regression)
3. **Strong baselines available**: "Tomorrow = today" persistence model
4. **Time series introduction**: Natural gateway to temporal ML concepts
5. **Interpretable metrics**: MAE in degrees Fahrenheit vs abstract accuracy percentages

**Expected Learning Outcomes**:
- **Problem type comparison**: Discrete vs continuous prediction
- **Evaluation differences**: Accuracy/confusion matrix vs MAE/RMSE/R²
- **Temporal dependencies**: How yesterday influences tomorrow
- **Baseline importance**: Weather persistence as strong competitor

**Implementation Plan**:
1. Start with persistence baseline (tomorrow = today)
2. Linear regression with multiple weather features
3. Neural network comparison (same architectural philosophy)
4. Feature engineering exploration (moving averages, seasonal trends)

**Next Session Goals**:
- Implement persistence baseline and dataset structure
- Create temporal train/test splits (respect time ordering)
- Build linear regression model for comparison

---

## 2025-07-30: L2 Regularization Implementation

**Context**: Adding regularization techniques to demonstrate overfitting prevention strategies

**Implementation Details**:
- Added L2 regularization (weight decay) parameter to SGD optimizer
- Implemented `--weight-decay` command-line argument with default 0.0
- Updated logging to track regularization strength alongside other hyperparameters

**Regularization Experiments**:
Tested three regularization strengths over 50 epochs:

1. **No Regularization** (weight_decay=0.0):
   - Test Accuracy: 75.8%
   - Final Training Loss: 0.5145, Validation Loss: 0.5198

2. **Moderate Regularization** (weight_decay=0.001):
   - Test Accuracy: 75.7%
   - Final Training Loss: 0.5166, Validation Loss: 0.5218

3. **Strong Regularization** (weight_decay=0.01):
   - Test Accuracy: 75.7%
   - Final Training Loss: 0.5273, Validation Loss: 0.5315

**Key Observations**:
- **Minimal overfitting detected**: All experiments showed training/validation losses converging
- **Regularization impact**: Higher weight decay slightly increased both training and validation losses
- **Performance consistency**: All configurations achieved ~75.7-75.8% accuracy
- **Pedagogical value**: Demonstrates that this problem doesn't require heavy regularization

**Why Minimal Regularization Impact?**
1. **Simple architecture**: Only 1 → 32 → 3 neurons (minimal parameters to overfit)
2. **Large dataset**: 56,520 temperature records provide ample training data
3. **Simple pattern**: Temperature-to-season mapping is relatively straightforward
4. **Early stopping implicit**: 50 epochs may prevent significant overfitting

**Educational Insights**:
- **Not all problems need regularization**: Demonstrates when simple models suffice
- **Regularization trade-offs**: Shows how weight decay affects training dynamics
- **Hyperparameter exploration**: Command-line interface enables easy experimentation
- **Baseline comparison remains crucial**: Temperature percentiles still match ML performance

**Technical Implementation**:
- L2 regularization added via PyTorch's built-in `weight_decay` parameter
- Clean integration with existing training pipeline
- Proper logging and experiment tracking maintained

---

## Next Steps & Open Questions

### Classification Project (Complete):
1. **✅ Overfitting diagnostics**: Successfully implemented with visual examples
2. **✅ Baseline reality check**: Discovered temperature percentiles match ML performance
3. **✅ L2 regularization**: Implemented and tested weight decay effects
4. **✅ Educational documentation**: Complete development log and insights captured

### Regression Project (Next Phase):
1. **Implement next-day temperature prediction**: Start with persistence baseline
2. **Temporal data handling**: Proper time series train/test splits
3. **Model comparison**: Linear regression vs neural network vs baseline
4. **Feature engineering**: Explore weather variable combinations and temporal features
5. **Educational contrast**: Document differences from classification problem

*Log maintained by: Claude Code assistant + Spencer*
*Purpose: Educational material development for ML fundamentals course*