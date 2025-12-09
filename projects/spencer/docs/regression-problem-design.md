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
| **Baseline** | Temperature percentiles (75.6%) | Climatology, Persistence, AR(1) |
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

### **Baseline Hierarchy & Expectations**

**Raw Temperature Baselines:**
1. **Climatology**: `temp_tomorrow = daily_normal[day_of_year]` 
   - Pure seasonal cycle, likely strongest baseline
   - Shows how much skill comes from just knowing the calendar
2. **Persistence**: `temp_tomorrow = temp_today`
   - Strong baseline for day-to-day continuity (~1°F typical change)
   - Works well even in transitional seasons due to gradual changes
3. **AR(1)**: `temp_tomorrow = α + β × temp_today`
   - Simple autoregressive model with mean reversion
   - May only marginally improve on persistence

**Temperature Anomaly Models:**
- **Climatology**: N/A (anomalies have zero climatology by definition)
- **Persistence**: `anomaly_tomorrow = anomaly_today` 
   - Reasonable baseline since weather patterns persist
- **AR(1)**: `anomaly_tomorrow = α + β × anomaly_today`
   - Should beat persistence through mean reversion to normal

**Expected Performance:** Simple statistical baselines likely hard to beat

### **Evaluation Strategy**
1. **Multiple Baselines**: Climatology → Persistence → AR(1) → ML models
2. **Raw vs Anomaly**: Compare problem formulations
3. **Metrics**: MAE (primary), RMSE (secondary), R² (goodness of fit)

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

### **Revised Expectations:**
- **Climatology**: Dominant baseline for raw temperatures (seasonal cycle)
- **Persistence**: Surprisingly strong for both raw temps and anomalies
- **AR(1)**: Modest improvement over persistence, may not beat climatology
- **ML models**: May struggle to beat simple statistical methods
- **Key insight**: Problem formulation (raw vs anomaly) affects baseline availability

### **Key Teaching Moments:**
1. **Climatology is powerful**: Most skill comes from seasonal cycle
2. **Day-to-day persistence**: Small daily temperature changes make persistence strong
3. **Problem formulation matters**: Raw vs anomaly changes baseline hierarchy
4. **Simple baselines**: AR(1) and persistence often compete with ML
5. **When ML doesn't help**: Some problems favor statistical approaches
6. **Evaluation differences**: "3°F MAE" vs "76% accuracy" interpretation

## Updated Implementation Plan

### **Phase 1: Foundation**
1. **Temporal data splitting** (crucial for time series validation)
2. **Climatology computation** (365-day normals from training data only)
3. **Data setup** for both raw temperatures and anomalies

### **Phase 2: Raw Temperature Models**
1. **Climatology baseline**: Daily normal temperatures
2. **Persistence baseline**: Tomorrow = today
3. **AR(1) model**: Simple autoregressive (no seasonal terms)
4. **Linear regression**: Multiple weather features
5. **Neural network**: Same architectural approach as classification

### **Phase 3: Anomaly Models** 
1. **Persistence**: Anomaly tomorrow = anomaly today
2. **AR(1)**: Pure autoregressive on anomalies
3. **Linear/NN models**: Focus on weather patterns without seasonality

### **Phase 4: Comparative Analysis**
- Raw vs anomaly model performance
- Statistical vs ML method comparison
- Skill attribution: seasonal cycle vs weather variability

## Success Criteria

**Good Student Learning Outcome**: Students understand
- Difference between classification and regression problems
- How to evaluate continuous predictions vs discrete ones
- Importance of temporal baselines in time series problems
- When simple models (linear regression) might outperform complex ones (neural networks)

---

*Decision documented: 2025-07-25*  
*Next: Implementation of next-day temperature prediction model*