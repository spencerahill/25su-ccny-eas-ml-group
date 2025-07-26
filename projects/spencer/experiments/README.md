# Experiments Directory

This directory tracks individual experiments and their results for pedagogical documentation.

## Structure

Each experiment should be documented with:
- **Parameters used** (learning rate, epochs, architecture, etc.)
- **Results achieved** (accuracy, loss curves, confusion matrices)
- **Plots generated** (training curves, confusion matrices)
- **Key observations** and lessons learned
- **Next experiment ideas**

## Naming Convention

- `YYYY-MM-DD_experiment-name/`
- Include all outputs (logs, plots, config files)
- Brief summary in each experiment folder

## Current Experiments

### 2025-07-25_baseline-comparison/
- **Goal**: Compare ML model vs temperature percentile baseline
- **Result**: Baseline 75.6% vs ML 75.7% - essentially equivalent!
- **Key Learning**: Simple domain knowledge can match complex models
- **Files**: 
  - `temp_season_classifier_20250725_230506.log`
  - `loss_curves_20250725_230549.png`
  - `accuracy_curves_20250725_230550.png`
  - `confusion_matrix_20250725_230550.png`

### 2025-07-25_overfitting-demo/
- **Goal**: Generate clear overfitting patterns for teaching
- **Parameters**: lr=0.1, epochs=200 (vs default lr=0.01, epochs=100)
- **Result**: Successfully induced overfitting - validation metrics became volatile
- **Key Learning**: Higher learning rates can promote overfitting for pedagogical demos
- **Files**:
  - `loss_curves_20250725_221445.png` - Shows clear train/val divergence
  - `accuracy_curves_20250725_221445.png` - Validation accuracy volatility

## Experiment Template

```markdown
# Experiment: [Name]

**Date**: YYYY-MM-DD
**Hypothesis**: What we expected to happen
**Parameters**: 
- Learning rate: X
- Epochs: Y
- Architecture: Z

**Results**:
- Accuracy: X%
- Key metrics: ...

**Observations**:
- What surprised us?
- What worked/didn't work?

**Next Steps**:
- What to try next based on these results

**Files Generated**:
- List of plots, logs, model files
```