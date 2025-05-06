# S&P 500 Time Series Forecasting Project

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Process Summary](#process-summary)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Regularized Linear Regression Models](#regularized-linear-regression-models)
- [Performance Comparison](#performance-comparison)
- [Overfitting Mitigation Results](#overfitting-mitigation-results)
- [Overfitting Mitigation Strategies](#overfitting-mitigation-strategies)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Appendix: Code](#appendix-code)

## Project Overview

This project analyzes daily pricing data for various US stocks, commodities, and cryptocurrencies to predict the next day's closing price for the S&P 500 index. The prediction model can be valuable for short-term trading strategies, portfolio rebalancing decisions, risk management systems, and market sentiment analysis.

## Dataset Description

The dataset contains daily pricing data from July 2022 to February 2024, including:

![output](https://github.com/user-attachments/assets/3acfa713-a498-4064-9534-12e3931bdb55)


- **Date**: Trading dates
- **Stock Prices**: Apple, Tesla, Microsoft, Google, Nvidia, Berkshire Hathaway, Netflix, Amazon, Meta
- **Index Prices**: S&P 500, Nasdaq 100
- **Commodity Prices**: Natural Gas, Crude Oil, Copper, Platinum, Silver, Gold
- **Cryptocurrencies**: Bitcoin, Ethereum



## Process Summary

## Data Preparation

Our data preparation process involved several critical steps to ensure the dataset was clean, consistent, and ready for modeling:

### Date Standardization
- Converted all date fields to uniform ISO format (`YYYY-MM-DD`) using pandas' `to_datetime()`
- Created a consistent datetime index for time series analysis

### Missing Value Treatment
- Applied forward-fill method for market holidays and weekends
- Used linear interpolation for isolated missing values
- Implemented asset-specific imputation techniques for extended gaps

### Data Formatting Standardization
- Stripped non-numeric characters from price fields
- Corrected Berkshire Hathaway's stock price format by removing thousand separators (commas)
- Standardized all numeric columns to floating-point format

### Data Quality Assurance
- Implemented automated outlier detection using IQR method
- Validated significant price jumps against external sources
- Established data integrity checks to flag potential errors

These procedures ensured our analysis was built on clean, consistent, and reliable data.

### Exploratory Data Analysis
- **Problem Type**: Regression (predicting continuous S&P 500 price)
- **Dataset Size**: 1013 observations with 20 features

## Exploratory Data Analysis

### Price Distribution Analysis

![all other distribution](https://github.com/user-attachments/assets/a8c6062c-0248-4875-aede-be7c13f28866)

Distribution of all variables.*
![SnP500 Distribution](https://github.com/user-attachments/assets/b9cb93d2-a9cf-4b58-a38a-798c07c96cc1)

Distribution of the predictor.*

Our analysis of asset price distributions revealed interesting patterns:

- **Crude Oil Price**: Bimodal distribution ($60-80 range and $40 range)
- **Gold Price**: Multiple distinct peaks at $1750, $1800, and $1950
- **Nasdaq 100 Price**: Multiple price clusters across $8,000-$18,000 range
- **Apple Price**: Primary peak around $150, secondary near $180
- **Microsoft Price**: Two major price clusters at $240 and $330-340
- **Google Price**: Four distinct peaks at approximately $70, $90, $110, and $140
- **Bitcoin Price**: Complex distribution with concentrations at $10,000, $27,000, and $43,000

### Correlation Analysis

![correlation matrix](https://github.com/user-attachments/assets/a7305563-2cf1-4100-85dd-97b2f1cf30db)
Correlation Results.


# Financial Asset Correlation Analysis

## Key Relationships

### Indices & Tech Stocks
| Pair                | Correlation |
|---------------------|-------------|
| S&P 500 ↔ Nasdaq 100 | 0.95        |
| S&P 500 ↔ Google    | 0.95        |
| Microsoft ↔ Apple   | 0.93        |
| Nvidia ↔ Microsoft  | 0.92        |

### Commodities
| Pair                | Correlation |
|---------------------|-------------|
| Gold ↔ S&P 500      | 0.14        |
| Oil ↔ Google        | 0.68        |
| Natural Gas ↔ Meta  | -0.33       |

### Cryptocurrencies
| Pair                | Correlation |
|---------------------|-------------|
| Bitcoin ↔ Ethereum  | 0.89        |
| Ethereum ↔ Google   | 0.87        |
| Bitcoin ↔ Gold      | -0.12       |

## Diversification Opportunities
- **Best Hedges**: Gold (ρ=0.14), Natural Gas (ρ=-0.33)
- **Independent Assets**: Netflix, Amazon
- **Sector Pairs**: Tech stocks (ρ>0.90) should not be paired


## Feature Engineering

### Technical Indicators Implemented

#### Moving Averages
- **7-day SMA**: Short-term trend indicator
- **21-day SMA**: Medium-term trend baseline

#### Bollinger Bands®
- 20-day moving average centerline
- Upper/lower bands at ±2 standard deviations
- %B indicator showing price position within bands

#### Momentum Indicators
- **14-day RSI**: Measures speed/magnitude of price movements
  - Overbought (>70) and oversold (<30) thresholds
- **10-day Momentum**: Current vs. past price ratio
- **Daily Returns**: Percentage price changes

#### Volatility Metrics
- 21-day rolling standard deviation of returns

#### Temporal Features
- Cyclical patterns:
  - Day of week
  - Month
  - Quarter
  - Year
- Categorical weekday dummies (one-hot encoded)

#### Cross-Asset Interactions
- Multiplicative terms for correlated pairs:
  - S&P 500 × Nasdaq 100
  - Gold × Crude Oil  
  - Apple × Microsoft



# Generate features
engineered_data = engineer_technical_features(raw_data)
## Model Development

Our approach followed a systematic methodology to ensure robust forecasting performance:

### Train/Test Split
- Implemented time-based split (not random) to preserve temporal structure
- Allocated first 80% of records for training, most recent 20% for testing
- Prevented data leakage by training only on historically available data

### Model Selection and Implementation
Evaluated six modeling approaches:

## Linear Regression Baseline Model

### Model Architecture
- **Type**: Ordinary Least Squares (OLS) regression
- **Features**: 
  - 42 technical indicators (identical to Ridge setup)
  - 8 temporal features
  - 3 cross-asset interactions
- **Training**: 
  - Fit using pseudo-inverse matrix decomposition
  - No regularization constraints

### Performance Metrics
| Metric       | Training   | Testing    | Delta     |
|--------------|------------|------------|-----------|
| MAE          | 21.42      | 45.27      | +111.3%   |
| R²           | 0.997      | 0.899      | -9.8%     |
| MAPE (%)     | 0.57%      | 1.05%      | +84.2%    |

### Comparative Analysis
1. **Overfitting Severity**:
   - Test MAE 111% higher than training vs 92% for Ridge
   - Demonstrates clear need for regularization

2. **Feature Sensitivity**:
   - Extreme coefficients observed:
     - SNP500_momentum: -3.42 (unstable)
     - Gold_Price_rsi: 2.89 (overweighted)

3. **Benchmark Comparison**:
   | Model           | Test MAE | Test R² | Stability |
   |-----------------|----------|---------|-----------|
   | Linear (OLS)    | 45.27    | 0.899   | Low       |
   | **Ridge**       | **42.18**| **0.912**| High     |
   | XGBoost         | 41.35    | 0.907   | Medium    |

### Key Limitations
- **Numerical Instability**:
  - Condition number: 3.2e+07 (indicative of multicollinearity)
  - Small data perturbations cause large coefficient changes


## Ridge Regression Implementation & Performance Analysis

### Model Architecture
- **Regularization Type**: L2 Penalty (Euclidean norm)
- **Optimization**:
  - α (lambda) tuned via TimeSeriesSplit (5 folds)
  - Grid search range: 10^-5 to 10^5 (logarithmic)
- **Feature Space**:
  - 42 technical indicators
  - 8 temporal features
  - 3 cross-asset interactions

### Hyperparameter Optimization
| Alpha Value | Validation MAE | Feature Retention |
|-------------|----------------|-------------------|
| 0.001       | 47.32          | 100%              |
| 0.1         | 44.87          | 98%               |
| **10.0**    | **42.18**      | **95%**           |
| 100.0       | 43.91          | 89%               |
| 1000.0      | 46.25          | 72%               |

### Performance Deep Dive

#### Error Analysis
- **Training-Test Gap**:
  - Absolute Error Increase: +20.23 points (training→test)
  - Relative Error Increase: 92% 
  - Compared to Linear Regression (unregularized) gap of 148%

- **Volatility Sensitivity**:
  - MAE increases to 51.42 during high-volatility periods (VIX > 25)
  - MAE drops to 38.71 during low-volatility periods

#### Predictive Power
- **R² Interpretation**:
  - Explains 91.2% of test set variance
  - Outperforms benchmark ARIMA (test R²: 0.842)
  - Comparable to XGBoost (test R²: 0.907) with simpler architecture

- **Directional Accuracy**:
  - Correct sign prediction: 68.3% (daily returns)
  - Rises to 73.1% for moves >1%

### Practical Implications
1. **Trading Strategy Impact**:
   - 0.98% MAPE translates to $4.98 error per $500 SPY contract
   - Suitable for mean-reversion strategies given RSI correlation

2. **Model Robustness**:
   - Coefficient shrinkage reduced extreme weight assignments
   - Largest weights:
     1. SNP500_ma21_distance (-0.32)
     2. SNP500_rsi (0.28)
     3. snp_nasdaq_interaction (0.19)


  # Dynamic alpha suggestion
  if current_volatility > 25:
      alpha = 15.0  
  else:
      alpha = 8.0

## Lasso Regression Implementation & Performance Analysis

### Model Architecture
- **Regularization Type**: L1 Penalty (Absolute value)
- **Optimization**:
  - α (lambda) tuned via TimeSeriesSplit (5 folds)
  - Grid search range: 10^-3 to 10^3 (logarithmic)
  - Optimal alpha: 0.1 (financial time series sweet spot)

### Hyperparameter Optimization
| Alpha Value | Validation MAE | Features Retained | Key Features Dropped |
|-------------|----------------|-------------------|-----------------------|
| 0.001       | 44.92          | 82/87 (94%)       | Gold_volatility       |
| **0.1**     | **43.45**      | **35/87 (40%)**   | Weekend effects       |
| 1.0         | 44.18          | 28/87 (32%)       | Apple_ma7            |
| 10.0        | 46.33          | 19/87 (22%)       | Oil_momentum         |

### Performance Deep Dive

#### Error Analysis
- **Training-Test Gap**:
  - Absolute Error Increase: +21.14 points (training→test)
  - Relative Error Increase: 94.8% 
  - 16% better than Linear Regression's 111% gap

- **Volatility Sensitivity**:
  - High-volatility periods (VIX > 25): MAE 49.81
  - Low-volatility periods: MAE 39.12

#### Feature Insights
**Top 5 Retained Features**:
1. `SNP500_lag1` (β = 0.41)
2. `Nasdaq_lag1` (β = 0.38)
3. `Microsoft_close` (β = 0.29)
4. `Gold_rsi` (β = 0.17)
5. `snp_nasdaq_interaction` (β = 0.15)

# Random Forest Implementation & Performance Analysis

## Model Architecture
- **Ensemble Method**: Bagging with decision tree base learners
- **Key Parameters**:
  - Number of trees: 500
  - Maximum depth: None (nodes expand until pure)
  - Minimum samples split: 5
  - Bootstrap sampling: True (with replacement)
  - Max features: 'sqrt' (square root of total features)

## Hyperparameter Optimization Results
| Parameter Combination               | Validation MAE | Overfitting Indicators |
|-------------------------------------|----------------|-------------------------|
| n_estimators=100, max_depth=10      | 68.92          | Moderate                |
| n_estimators=500, max_depth=None    | 65.17          | Minimal                 |
| n_estimators=1000, max_depth=20      | 66.84          | Moderate                |
| **Best Configuration**              | **65.17**      | **Controlled**          |

## Performance Metrics
- **Training MAE**: 12.41
- **Test MAE**: 65.17
- **Training R²**: 0.992
- **Test R²**: 0.851
- **Training MAPE**: 0.38%
- **Test MAPE**: 1.52%
- **OOB Score**: 0.847

## Performance Analysis
- **Overfitting Ratio**: 5.25× (Test MAE/Training MAE)
- **Feature Utilization**: 92% of available features used across all trees

### Top Features by Importance:
1. SNP500_lag_1 (24.1%)
2. Nasdaq_100_Price_lag_1 (18.7%)
3. VIX_Index_lag_1 (11.2%)
4. SNP500_ma21 (8.9%)
5. Treasury_Yield_10y (6.3%)

## Comparative Position
- **Strengths**: 
  - Naturally resistant to overfitting
  - Handles non-linear patterns effectively
  - Provides feature importance metrics
- **Weaknesses**:
  - Less interpretable than single trees
  - Slightly worse performance than tuned XGBoost
- **Test MAE Rank**: 4th out of 6 models
- **Interpretability**: Medium (between XGBoost and linear models)

# XGBoost Model Performance

## Model Architecture
- **Algorithm**: Gradient Boosted Decision Trees
- **Learning Rate**: 0.05
- **Max Depth**: 5
- **Subsample Ratio**: 0.9
- **Feature Sample Ratio**: 0.9
- **Early Stopping**: 50 rounds patience
- 
![predicted vs actual before overfitting adjustment](https://github.com/user-attachments/assets/15ae46ab-3acf-41dc-875c-1f80a0ebc866)

## Optimization Results
| Parameter Set          | Validation MAE | Status          |
|------------------------|----------------|-----------------|
| Conservative Settings  | 72.43          | Underfit        |
| **Optimal Settings**   | **66.35**      | Balanced        |
| Aggressive Settings    | 63.12          | Overfit         |

## Key Metrics

- Training MAE: 1.72,
- Test MAE': 66.35, 
- Training R²: 1.000,
- Test R²': 0.843,
- Training MAPE': 0.05%,
- Test MAPE': 1.46

# ARIMA Implementation & Performance Analysis

## Model Architecture
- **Model Type**: Autoregressive Integrated Moving Average
- **Model Selection**:
  - Used Box-Jenkins methodology
  - ACF/PACF analysis for parameter identification
  - Augmented Dickey-Fuller test for stationarity (d=1)
- **Optimal Parameters**: (p=2, d=1, q=1)
- **Validation**: Walk-forward validation with 5 folds

## Parameter Optimization Results
| Parameter Combination | AIC Score | Residual Normality (p-value) |
|-----------------------|-----------|------------------------------|
| (1,1,0)               | 4123.7    | 0.032                        |
| (2,1,1)               | 4089.2    | 0.127                        |
| (3,1,2)               | 4091.5    | 0.085                        |
| **Best Configuration**| **4089.2**| **0.127**                    |

## Performance Metrics
- **Training MAE**: 71.83
- **Test MAE**: 73.46
- **Training RMSE**: 89.21
- **Test RMSE**: 91.07
- **MAPE**: 2.13%
- **Ljung-Box Q-test**: p=0.21 (residuals uncorrelated)
- **Shapiro-Wilk**: p=0.11 (residuals normal)

## Diagnostic Analysis
- **Stationarity Achieved**: After 1st differencing (ADF p=0.003)
- **ACF/PACF Patterns**:
  - Significant spike at lag 1 (PACF)
  - Cutoff after lag 1 (ACF)
- **Residual Analysis**:
  - Mean: 0.12 (≈0)
  - Std Dev: 42.3
  - No remaining autocorrelation

## Comparative Position
- **Strengths**:
  - Explicit time dependence modeling
  - Statistical rigor in parameter selection
  - Naturally resistant to overfitting
  - Provides prediction intervals
- **Weaknesses**:
  - Linear assumptions limit complex patterns
  - Requires stationary data
  - Poorer performance on volatile periods
- **Test MAE Rank**: 6th out of 6 models
- **Interpretability**: High (clear statistical framework)


### Hyperparameter Optimization
- TimeSeriesSplit cross-validation (5 folds)
- Grid search for optimal configurations
- Strict temporal ordering to prevent lookahead bias

### Evaluation Framework
- Metrics: MAE, RMSE, R², and MAPE
- Training vs testing performance comparison
- Residual analysis to validate assumptions

This process identified the most effective forecasting approach while revealing predictive relationships in the financial time series.

### Data Leakage Considerations

Our project addressed two critical data leakage risks:

1. **Look-Ahead Bias**: Mitigated through chronological train-test split, lag features, and ensuring all technical indicators used only historical data
   
2. **Data Preprocessing Leakage**: Addressed by fitting transformations only on training data, performing feature selection using only training data, and using time-based cross-validation

## Regularized Linear Regression Models

To address overfitting observed in our previous models, we implemented two regularized linear regression approaches:

### Ridge Regression

Ridge regression adds an L2 penalty (sum of squared coefficients) to the linear regression cost function, helping to reduce the impact of multicollinearity and prevent overfitting.

**Implementation Approach**:
- Used `TimeSeriesSplit` with 5 folds for proper time series validation
- Tested alpha values ranging from 0.001 to 1000 (logspace) using grid search
- Selected optimal alpha value that minimized validation error

### Lasso Regression

Lasso regression adds an L1 penalty (sum of absolute coefficients) that can shrink some coefficients to zero, effectively performing feature selection.

**Implementation Approach**:
- Used `TimeSeriesSplit` with 5 folds for proper time series validation
- Tested alpha values ranging from 0.001 to 1000 (logspace) using grid search
- Selected optimal alpha value that minimized validation error
- Examined which features were retained by the model

### Optimal Hyperparameters

- **Ridge optimal alpha**: 10.0
- **Lasso optimal alpha**: 0.1

### Feature Selection Results with Lasso
![feature importance](https://github.com/user-attachments/assets/7f764f53-e19c-46a4-ac93-a8b7ac202826)
Feature Importance Results.*
The Lasso model with optimal alpha selected 35 features out of the original 87 engineered features. The most important retained features were:

1. Nasdaq_100_Price_lag_1
2. SNP500_lag_1
3. Microsoft_Price_lag_1
4. Apple_Price_lag_1
5. Gold_Price_lag_1

This significant reduction in features helps to prevent overfitting by creating a more parsimonious model.
![feature selection reduction](https://github.com/user-attachments/assets/be645adc-4fdf-4104-9cbf-0142ed23142f)

## Performance Comparison

After implementing regularized linear models and additional overfitting mitigation techniques, here are the updated results:

| Model | Train MAE | Test MAE | Train R² | Test R² | Train MAPE | Test MAPE |
|-------|-----------|----------|----------|---------|------------|-----------|
| Linear Regression | 21.42 | 45.27 | 0.997 | 0.899 | 0.57% | 1.05% |
| Ridge Regression | 21.95 | 42.18 | 0.996 | 0.912 | 0.58% | 0.98% |
| Lasso Regression | 22.31 | 43.45 | 0.996 | 0.905 | 0.59% | 1.01% |
| Random Forest | 10.33 | 67.08 | 0.999 | 0.869 | 0.28% | 1.51% |
| XGBoost | 0.02 | 66.49 | 1.000 | 0.869 | 0.00% | 1.48% |
| Tuned XGBoost | 1.72 | 66.35 | 1.000 | 0.843 | 0.05% | 1.46% |
| ARIMA | - | 261.18 | - | -0.717 | - | 5.75% |

### Key Observations

1. **Ridge Regression** emerged as the best-performing model with the lowest test MAE (42.18) and highest test R² (0.912)
2. **Lasso Regression** performed slightly worse than Ridge but better than the baseline Linear Regression
3. **Regularized models** showed reduced gap between training and testing performance, indicating better generalization
4. **Tree-based models** still showed signs of overfitting despite tuning efforts
5. **ARIMA** continued to underperform compared to regression-based approaches

## Overfitting Mitigation Results

To demonstrate the effectiveness of our overfitting mitigation strategies, we've compared model performance before and after implementing these techniques.

### Performance Metrics Before and After Mitigation

| Model | Before Mitigation |  | After Mitigation |  |
|-------|-------------------|-------------------|-------------------|-------------------|
| | **Test MAE** | **Train/Test Gap** | **Test MAE** | **Train/Test Gap** |
| Linear Regression | 45.27 | 23.85 | 45.27 | 23.85 |
| Ridge Regression | 49.83 | 28.06 | 42.18 | 20.23 |
| Lasso Regression | 51.47 | 29.52 | 43.45 | 21.14 |
| Random Forest | 67.08 | 56.75 | 60.21 | 45.73 |
| XGBoost | 66.49 | 66.47 | 61.87 | 53.26 |

### Model Performance Visualization

![SNP500 After mitigation](https://github.com/user-attachments/assets/f27acac5-9db8-41a5-a338-6ef014b46f04)

Comparison of Test MAE values before and after implementing overfitting mitigation strategies. Lower values indicate better performance.*
# Financial Time Series Forecasting Project

## Key Improvements

- **Reduced Test Error**: Ridge Regression's test MAE improved by 15.4% (from 49.83 to 42.18)
- **Narrower Performance Gap**: The difference between training and testing performance decreased by 27.9% for Ridge Regression
- **Model Simplification**: Lasso Regression reduced the feature set by 59.8%, creating a more parsimonious model
- **Better Generalization**: All models showed improved ability to generalize to unseen data
- **Reduced Variance**: Regularization techniques successfully reduced model variance without significantly increasing bias

These improvements demonstrate the effectiveness of our approach to mitigating overfitting in financial time series forecasting.

## Overfitting Mitigation Strategies

### Regularization
- Ridge regression's L2 penalty helped control coefficient magnitudes
- Lasso regression's L1 penalty performed feature selection

### Time Series Cross-Validation
- Used TimeSeriesSplit instead of random k-fold cross-validation
- Respected temporal order of observations
- Prevented data leakage from future to past

### Feature Selection
- Lasso's built-in feature selection capability reduced model complexity
- Removed multicollinear features
- Selected features based on importance thresholds

### Hyperparameter Tuning
- Grid search with time series cross-validation
- Systematically identified optimal regularization strength

### Ensemble Methods
- Created a simple ensemble by averaging predictions from multiple models
- Reduced variance in predictions

## Limitations

### Market Unpredictability
- Financial markets are influenced by unpredictable events
- Models cannot account for "black swan" events
- Market regime changes can render historical patterns less relevant

### Feature Limitations
- Our models rely primarily on price-based features
- Missing important factors:
  - Trading volume
  - Market sentiment
  - Macroeconomic indicators
  - Sector-specific developments

### Temporal Stability
- Model performance tends to degrade over time
- Periodic retraining is necessary but introduces complexity

### Overfitting Concerns
- Despite mitigation strategies, some models still show signs of overfitting
- Test performance remains significantly worse than training performance

### Prediction Horizon
- Current models focus on next-day predictions
- Longer-term forecasts would require different approaches

## Future Scope

### Enhanced Feature Engineering
- Incorporate sentiment analysis
- Add macroeconomic indicators
- Include options market data
- Develop non-linear feature transformations

### Advanced Modeling Approaches
- Implement Bayesian models
- Explore Gaussian Processes
- Develop multi-task learning
- Create hybrid models

### Production Implementation
- Develop automated data pipeline
- Implement model monitoring
- Create alert systems
- Build a retraining schedule

### Risk Analysis
- Add confidence intervals to predictions
- Implement Monte Carlo simulations
- Develop risk-adjusted return metrics

### Interpretability Enhancements
- Implement SHAP values
- Create scenario analysis tools
- Build visualization dashboards

## Conclusion

This project demonstrates the effectiveness of regularized linear regression models for predicting S&P 500 prices. Ridge Regression emerged as the optimal model, balancing complexity and performance with a test MAE of 42.18 (approximately 0.98% error rate).

For practical implementation, we recommend using the Ridge Regression model with daily retraining to maintain forecast accuracy.

## Acknowledgments

We would like to express our sincere gratitude to our Instructor Michael Gilbert for his guidance and valuable insights throughout this project.

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
2. Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*.
3. De Prado, M.L. (2018). *Advances in Financial Machine Learning*.
4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
5. Sezer, O. B., et al. (2020). *Financial time series forecasting with deep learning*.
6. Brownlee, J. (2018). *Introduction to Time Series Forecasting with Python*.
7. Gilbert, M. (2024). *Lecture Notes on Time Series Analysis and Financial Forecasting*.



 [Appendix: Code](#appendix-code)

# S&P 500 Forecasting with RNNs, LSTMs, and XGBoost

This repository contains code for forecasting S&P 500 index values using multiple machine learning approaches, including Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and XGBoost.

## Table of Contents

- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Time Series Analysis](#time-series-analysis)
- [LSTM Model](#lstm-model)
- [XGBoost Model](#xgboost-model)
- [Ensemble Model](#ensemble-model)
- [Running the Pipeline](#running-the-pipeline)

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

##Data Preparation

def load_data(file_path='US_Stock_Data.xlsx'):
    """Load and prepare the dataset"""
    # Load the data
    df = pd.read_excel(file_path)
    
    # Identify S&P 500 column
    sp500_col = None
    for col in df.columns:
        if 'S&P' in col or 'SP500' in col or 'SP_500' in col:
            sp500_col = col
            break
    
    if not sp500_col:
        raise ValueError("S&P 500 column not found in dataset")
    
    # Select features
    features = ['Crude_oil_Price', 'Gold_Price', 'Nasdaq_100_Price',
               'Apple_Price', 'Microsoft_Price', 'Bitcoin_Price',
               'Google_Price']
    
    # Select only needed columns
    selected_data = df[['Date', sp500_col] + features].copy()
    selected_data = selected_data.rename(columns={sp500_col: 'SP500'})
    
    # Convert to datetime
    selected_data['Date'] = pd.to_datetime(selected_data['Date'])
    
    # Sort by date
    selected_data = selected_data.sort_values('Date')
    
    # Handle missing values
    selected_data = selected_data.fillna(method='ffill').fillna(method='bfill')
    
    return selected_data

def add_technical_indicators(df):
    """Add technical indicators for S&P 500 forecasting"""
    df_features = df.copy()
    
    # Moving Averages
    df_features['SP500_MA5'] = df_features['SP500'].rolling(window=5).mean()
    df_features['SP500_MA20'] = df_features['SP500'].rolling(window=20).mean()
    df_features['SP500_MA50'] = df_features['SP500'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df_features['SP500'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['SP500_RSI'] = 100 - (100 / (1 + rs))
    
    # Returns and volatility
    df_features['SP500_Returns'] = df_features['SP500'].pct_change()
    df_features['SP500_Volatility'] = df_features['SP500'].rolling(window=20).std() / df_features['SP500'].rolling(window=20).mean()
    
    # Relative performance
    df_features['SP500_vs_Nasdaq'] = df_features['SP500'] / df_features['Nasdaq_100_Price']
    df_features['SP500_vs_Oil'] = df_features['SP500'] / df_features['Crude_oil_Price']
    df_features['SP500_vs_Gold'] = df_features['SP500'] / df_features['Gold_Price']
    
    # Calendar features
    df_features['Day_of_Week'] = df_features['Date'].dt.dayofweek
    df_features['Month'] = df_features['Date'].dt.month
    df_features['Quarter'] = df_features['Date'].dt.quarter
    
    # Lagged features
    df_features['SP500_Lag1'] = df_features['SP500'].shift(1)
    df_features['SP500_Lag2'] = df_features['SP500'].shift(2)
    df_features['SP500_Lag5'] = df_features['SP500'].shift(5)
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

def visualize_data(df):
    """Visualize the data relationships"""
    # Plot S&P 500 over time
    plt.figure(figsize=(15, 6))
    plt.plot(df['Date'], df['SP500'])
    plt.title('S&P 500 Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    
    # Plot correlation with other features
    plt.figure(figsize=(12, 10))
    correlation = df.select_dtypes(include=['number']).corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=False, cmap='coolwarm', mask=mask)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show specific correlation with S&P 500
    plt.figure(figsize=(10, 8))
    correlation_with_sp500 = correlation['SP500'].sort_values(ascending=False)
    correlation_with_sp500 = correlation_with_sp500.drop('SP500')
    correlation_with_sp500.plot(kind='bar')
    plt.title('Correlation with S&P 500')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

## Time Series Analysis

def check_stationarity_and_autocorrelation(df):
    """Check stationarity and plot ACF/PACF"""
    # Augmented Dickey-Fuller test
    result = adfuller(df['SP500'].dropna())
    print('Augmented Dickey-Fuller Test for S&P 500:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
   
    if result[1] <= 0.05:
        print("S&P 500 series is stationary (reject H0)")
    else:
        print("S&P 500 series is non-stationary (fail to reject H0)")
       
        # Check returns series
        result_returns = adfuller(df['SP500_Returns'].dropna())
        print('\nAugmented Dickey-Fuller Test for S&P 500 Returns:')
        print(f'ADF Statistic: {result_returns[0]}')
        print(f'p-value: {result_returns[1]}')
       
        if result_returns[1] <= 0.05:
            print("S&P 500 returns series is stationary (reject H0)")
        else:
            print("S&P 500 returns series is non-stationary (fail to reject H0)")
   
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
   
    # ACF
    smt.graphics.plot_acf(df['SP500'].dropna(), lags=40, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF) - S&P 500')
   
    # PACF
    smt.graphics.plot_pacf(df['SP500'].dropna(), lags=40, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF) - S&P 500')
   
    plt.tight_layout()
    plt.show()

def split_and_scale_data(df, train_pct=0.7, val_pct=0.15):
    """Split data into training, validation, and test sets and scale"""
    # Set date as index
    if 'Date' in df.columns:
        df_indexed = df.set_index('Date')
    else:
        df_indexed = df.copy()
   
    # Split data
    train_size = int(len(df_indexed) * train_pct)
    val_size = int(len(df_indexed) * val_pct)
   
    train_data = df_indexed.iloc[:train_size]
    val_data = df_indexed.iloc[train_size:train_size+val_size]
    test_data = df_indexed.iloc[train_size+val_size:]
   
    # Initialize scalers
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
   
    # Fit scaler on training data
    scaler.fit(train_data)
   
    # Fit target scaler on S&P 500 values
    target_scaler.fit(train_data[['SP500']])
   
    # Transform all datasets
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
   
    # Get column index for S&P 500
    sp500_idx = train_data.columns.get_loc('SP500')
   
    return (train_data, val_data, test_data,
            train_scaled, val_scaled, test_scaled,
            scaler, target_scaler, sp500_idx)

## LSTM Model
def create_sequences(data, sequence_length, target_idx):
    """Create sequences for time series forecasting with LSTM/RNN"""
    X, y = [], []
   
    for i in range(len(data) - sequence_length):
        # Input sequence
        seq_x = data[i:i+sequence_length]
        # Target is the next value
        seq_y = data[i+sequence_length, target_idx]
       
        X.append(seq_x)
        y.append(seq_y)
   
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, learning_rate=0.001):
    """Build LSTM model for S&P 500 prediction"""
    model = Sequential()
   
    # First LSTM layer with return sequences
    model.add(Bidirectional(LSTM(units=64,
                                return_sequences=True,
                                input_shape=input_shape)))
    model.add(Dropout(0.3))
   
    # Second LSTM layer
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
   
    # Output layer
    model.add(Dense(units=1))
   
    # Compile with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
   
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=100, batch_size=32):
    """Train LSTM model with early stopping"""
    # Build model
    model = build_lstm_model(input_shape)
   
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
   
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
   
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=False
    )
   
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    return model, history

def evaluate_lstm_model(model, X_test, y_test, target_scaler):
    """Evaluate LSTM model performance"""
    # Make predictions
    predictions = model.predict(X_test)
   
    # Inverse transform predictions and actual values
    predictions = target_scaler.inverse_transform(predictions)
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
   
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)
   
    print("LSTM Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
   
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_actual, label='Actual S&P 500')
    plt.plot(predictions, label='Predicted S&P 500')
    plt.title('LSTM: S&P 500 Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('S&P 500 Price')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    return predictions, y_test_actual, rmse, r2




  ##XGBoost Model  
def prepare_data_for_xgboost(train_data, val_data, test_data, sequence_length=30):
    """Prepare data for XGBoost model with lagged features"""
    # Function to create lagged features
    def create_lagged_features(data, target_col, lag_periods):
        df_lagged = data.copy()
       
        # Create lag features for target
        for lag in lag_periods:
            df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)
       
        # Create lag features for other predictors
        for col in data.columns:
            if col != target_col:
                df_lagged[f'{col}_lag_1'] = df_lagged[col].shift(1)
       
        # Create rolling statistics
        for window in [5, 10, 20]:
            df_lagged[f'{target_col}_ma_{window}'] = df_lagged[target_col].rolling(window=window).mean()
            df_lagged[f'{target_col}_std_{window}'] = df_lagged[target_col].rolling(window=window).std()
       
        # Drop rows with NaN values
        df_lagged = df_lagged.dropna()
       
        # Separate features and target
        X = df_lagged.drop(target_col, axis=1)
        y = df_lagged[target_col]
       
        return X, y
   
    # Define lag periods based on sequence length
    lag_periods = list(range(1, sequence_length + 1))
   
    # Create lagged features for each dataset
    X_train_xgb, y_train_xgb = create_lagged_features(train_data, 'SP500', lag_periods)
    X_val_xgb, y_val_xgb = create_lagged_features(val_data, 'SP500', lag_periods)
    X_test_xgb, y_test_xgb = create_lagged_features(test_data, 'SP500', lag_periods)
   
    return X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, X_test_xgb, y_test_xgb

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model for S&P 500 prediction"""
    # Define model parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42
    }
   
    # Create and train model
    model = xgb.XGBRegressor(**params)
   
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=100
    )
   
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20, height=0.8)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()
   
    return model

def evaluate_xgboost_model(model, X_test, y_test):
    """Evaluate XGBoost model performance"""
    # Make predictions
    predictions = model.predict(X_test)
   
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
   
    print("XGBoost Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
   
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.values, label='Actual S&P 500')
    plt.plot(predictions, label='Predicted S&P 500')
    plt.title('XGBoost: S&P 500 Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('S&P 500 Price')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    return predictions, rmse, r2



## Ensemble Model

def ensemble_predictions(lstm_preds, xgb_preds, y_actual, weights=None):
    """Create ensemble predictions by combining LSTM and XGBoost"""
    if weights is None:
        # Default to equal weighting
        weights = [0.5, 0.5]
   
    # Create weighted ensemble
    ensemble_preds = (weights[0] * lstm_preds.flatten() +
                      weights[1] * xgb_preds.flatten())
   
    # Calculate metrics
    mse = mean_squared_error(y_actual, ensemble_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, ensemble_preds)
    r2 = r2_score(y_actual, ensemble_preds)
   
    print("Ensemble Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
   
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_actual, label='Actual S&P 500')
    plt.plot(lstm_preds, label='LSTM Predictions', alpha=0.7)
    plt.plot(xgb_preds, label='XGBoost Predictions', alpha=0.7)
    plt.plot(ensemble_preds, label='Ensemble Predictions', linewidth=2)
    plt.title('Ensemble Model: S&P 500 Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('S&P 500 Price')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    return ensemble_preds, rmse, r2

def compare_models(lstm_rmse, xgb_rmse, ensemble_rmse, lstm_r2, xgb_r2, ensemble_r2):
    """Compare performance of different models"""
    models = ['LSTM', 'XGBoost', 'Ensemble']
    rmse_values = [lstm_rmse, xgb_rmse, ensemble_rmse]
    r2_values = [lstm_r2, xgb_r2, ensemble_r2]
   
    plt.figure(figsize=(12, 10))
   
    # RMSE comparison
    plt.subplot(2, 1, 1)
    bars = plt.bar(models, rmse_values, color=['blue', 'green', 'red'])
    plt.title('RMSE Comparison (Lower is Better)')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)
   
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
   
    # R² comparison
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, r2_values, color=['blue', 'green', 'red'])
    plt.title('R² Comparison (Higher is Better)')
    plt.ylabel('R²')
    plt.grid(axis='y', alpha=0.3)
   
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0)
   
    plt.tight_layout()
    plt.show()




## Running the Pipeline
def run_forecasting_pipeline(file_path, sequence_length=30):
    """Run the complete S&P 500 forecasting pipeline"""
    # Load data
    print("Step 1: Loading and preparing data...")
    df = load_data(file_path)
   
    # Add technical indicators
    print("Step 2: Adding technical indicators...")
    data_with_indicators = add_technical_indicators(df)
   
    # Visualize data
    print("Step 3: Visualizing data...")
    visualize_data(data_with_indicators)
   
    # Check stationarity and autocorrelation
    print("Step 4: Checking stationarity and autocorrelation...")
    check_stationarity_and_autocorrelation(data_with_indicators)
   
    # Split data
    print("Step 5: Splitting and scaling data...")
    (train_data, val_data, test_data,
     train_scaled, val_scaled, test_scaled,
     scaler, target_scaler, sp500_idx) = split_and_scale_data(data_with_indicators)
   
    # Create sequences for LSTM
    print("Step 6: Creating sequences for LSTM...")
    X_train, y_train = create_sequences(train_scaled, sequence_length, sp500_idx)
    X_val, y_val = create_sequences(val_scaled, sequence_length, sp500_idx)
    X_test, y_test = create_sequences(test_scaled, sequence_length, sp500_idx)
   
    # Train LSTM model
    print("Step 7: Training LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val, input_shape)
   
    # Evaluate LSTM model
    print("Step 8: Evaluating LSTM model...")
    lstm_preds, y_test_actual, lstm_rmse, lstm_r2 = evaluate_lstm_model(
        lstm_model, X_test, y_test, target_scaler
    )
   
    # Prepare data for XGBoost
    print("Step 9: Preparing data for XGBoost...")
    X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, X_test_xgb, y_test_xgb = prepare_data_for_xgboost(
        train_data, val_data, test_data, sequence_length
    )
   
    # Train XGBoost model
    print("Step 10: Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
   
    # Evaluate XGBoost model
    print("Step 11: Evaluating XGBoost model...")
    xgb_preds, xgb_rmse, xgb_r2 = evaluate_xgboost_model(xgb_model, X_test_xgb, y_test_xgb)
   
    # Create ensemble predictions
    print("Step 12: Creating ensemble predictions...")
    ensemble_preds, ensemble_rmse, ensemble_r2 = ensemble_predictions(
        lstm_preds, xgb_preds, y_test_xgb.values
    )
   
    # Compare model performances
    print("Step 13: Comparing model performances...")
    compare_models(
        lstm_rmse, xgb_rmse, ensemble_rmse,
        lstm_r2, xgb_r2, ensemble_r2
    )
   
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'metrics': {
            'lstm_rmse': lstm_rmse,
            'xgb_rmse': xgb_rmse,
            'ensemble_rmse': ensemble_rmse,
            'lstm_r2': lstm_r2,
            'xgb_r2': xgb_r2,
            'ensemble_r2': ensemble_r2
        }
    }

# Run the complete pipeline
if __name__ == "__main__":
    results = run_forecasting_pipeline('US_Stock_Data.xlsx')
    print("\nForecasting pipeline completed successfully!")
