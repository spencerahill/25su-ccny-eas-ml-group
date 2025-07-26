# Experiment: Overfitting Pattern Generation

**Date**: 2025-07-25  
**Hypothesis**: Higher learning rate and more epochs will induce overfitting patterns for teaching

**Parameters**: 
- Learning rate: 0.1 (increased from 0.01)
- Epochs: 200 (increased from 100)
- Architecture: Same simple feedforward (1 → 32 → 3)

**Results**:
- Successfully generated clear overfitting patterns!
- Training loss continued decreasing while validation became volatile
- Perfect visual example for teaching overfitting concepts

**Key Observations**:
- **Loss curves**: Training loss steadily decreases, validation loss becomes erratic after epoch ~30
- **Accuracy curves**: Training accuracy more stable, validation accuracy highly volatile
- **Pattern emergence**: Classic "hockey stick" overfitting signature visible

**Teaching Value**:
These plots demonstrate exactly what students should look for:
1. **Early training** (epochs 1-20): Both metrics improve together (healthy)
2. **Overfitting onset** (epochs 20-40): Validation metrics become unstable  
3. **Clear overfitting** (epochs 40-200): Training improves while validation stagnates/oscillates

**Technical Insight**:
Higher learning rates can promote overfitting by:
- Making larger parameter updates each step
- Creating less stable convergence
- Enabling faster "memorization" of training data
- Increasing volatility in validation performance

**Next Steps**:
- Use these plots as canonical examples in teaching materials
- Experiment with regularization techniques (weight decay, dropout)
- Show how early stopping could prevent overfitting

**Files Generated**:
- `loss_curves_20250725_221445.png` - Clear train/validation divergence pattern
- `accuracy_curves_20250725_221445.png` - Validation accuracy volatility example