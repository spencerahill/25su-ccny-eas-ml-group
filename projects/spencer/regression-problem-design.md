# Regression Problem Design: Next-Day Temperature Prediction

*Documentation of design decisions for the regression component of Spencer's pedagogical ML project*

## Problem Statement

**Objective**: Predict tomorrow's average temperature using today's weather conditions
**Type**: Regression (continuous target variable)
**Target Variable**: `temp_avg[t+1]` (next day's average temperature in °F)

## Rationale for This Choice

### **Pedagogical Benefits**
1. **Immediately Relatable**: Students understand weather forecasting - it's tangible and relevant
2. **Clear Success Metric**: Mean Absolute Error in °F is interpretable ("off by 3 degrees")
3. **Natural Baselines**: "Tomorrow = Today" persistence model for comparison
4. **Real-world Application**: Actual weather forecasting problem with practical value

### **Technical Learning Opportunities**
1. **Regression vs Classification**: Direct comparison with season classification using same dataset
2. **Time Series Concepts**: Introduces temporal dependencies and autocorrelation
3. **Feature Engineering**: Multiple weather variables, seasonal trends, moving averages
4. **Model Evaluation**: Different metrics (MAE, RMSE) vs accuracy for classification

### **Dataset Advantages**
- **Rich feature set**: Daily min/max temperature, precipitation, snowfall, seasonal patterns
- **Long time series**: 150+ years allows for robust training and validation
- **Clean problem setup**: Well-defined temporal structure (predict t+1 from t)

## Problem Comparison: Classification vs Regression

| Aspect | Season Classification | Next-Day Temperature |
|--------|----------------------|---------------------|
| **Target** | Categorical (Winter/Summer/Spring-Fall) | Continuous (temperature in °F) |
| **Baseline** | Temperature percentiles (75.6%) | Persistence: tomorrow = today |
| **Evaluation** | Accuracy, confusion matrix | MAE, RMSE, R² |
| **Interpretability** | "Correct season 76% of time" | "Off by X degrees on average" |
| **Complexity** | Static pattern recognition | Temporal sequence prediction |

## Proposed Implementation Approach

### **Phase 1: Simple Regression**
```python
# Features: Today's weather conditions
X = [temp_avg_today, temp_min_today, temp_max_today, precip_today, day_of_year]
y = temp_avg_tomorrow

# Models to compare:
# 1. Linear regression
# 2. Simple neural network  
# 3. Persistence baseline
```

### **Phase 2: Enhanced Features** (if time permits)
```python
# Extended features: Recent weather history
X = [last_3_days_avg_temp, temp_range_today, precip_yesterday, 
     seasonal_indicators, moving_averages]
y = temp_avg_tomorrow
```

### **Evaluation Strategy**
1. **Persistence Baseline**: `temp_tomorrow = temp_today`
2. **Linear Regression**: Standard sklearn implementation
3. **Neural Network**: Similar architecture to classification model
4. **Metrics**: MAE (primary), RMSE (secondary), R² (goodness of fit)

## Educational Value Comparison

### **What Classification Problem Teaches:**
- Model evaluation with discrete outcomes
- Confusion matrices and classification metrics
- Overfitting detection with accuracy curves
- When ML doesn't help (baseline reality check)

### **What Regression Problem Adds:**
- Continuous prediction and error analysis
- Different evaluation metrics and their meanings
- Time series / temporal prediction concepts
- Feature engineering with multiple correlated variables

## Expected Outcomes & Lessons

### **Likely Results:**
- **Persistence baseline**: Probably quite strong (weather has momentum)
- **Linear regression**: May capture seasonal trends better than persistence
- **Neural network**: Might overfit given the simplicity of the relationship

### **Key Teaching Moments:**
1. **Weather has memory**: Yesterday's temperature is highly predictive
2. **Seasonal patterns**: Day-of-year as crucial feature
3. **Evaluation differences**: What does "3°F MAE" mean vs "76% accuracy"?
4. **Problem complexity**: Time series harder than static classification

## Next Steps

1. **Implement persistence baseline** first (establish benchmark)
2. **Create temporal dataset** with proper t → t+1 structure
3. **Implement linear regression** model with multiple features
4. **Compare to neural network** version (same architecture philosophy)
5. **Analyze temporal patterns** and model errors

## Success Criteria

**Good Student Learning Outcome**: Students understand
- Difference between classification and regression problems
- How to evaluate continuous predictions vs discrete ones
- Importance of temporal baselines in time series problems
- When simple models (linear regression) might outperform complex ones (neural networks)

---

*Decision documented: 2025-07-25*  
*Next: Implementation of next-day temperature prediction model*