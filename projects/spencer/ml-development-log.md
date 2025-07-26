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

## Next Steps & Open Questions

1. **Architecture experiments**: Try deeper networks, different activation functions
2. **Feature engineering**: Add seasonal indicators, moving averages, day-of-year
3. **Alternative baselines**: Calendar-based rules, climatological averages
4. **Problem reframing**: Regression instead of classification? Different season definitions?
5. **Data analysis**: Why is the simple baseline so strong? What patterns exist?

*Log maintained by: Claude Code assistant + Spencer*
*Purpose: Educational material development for ML fundamentals course*