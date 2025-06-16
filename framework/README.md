# Framework
This Dash-based framework is designed to impute, analyze and forecast building energy consumption time series data.

## Features
- Upload your own building energy time series dataset (in CSV format)
- Imputation Techniques
    - Last Observation Carried Forward (LOCF)
    - Next Observation Carried Backwards (NOCB)
    - Linear Interpolation
    - Linear Regression
    - LightGBM
- Evaluation
    - Error metrics available
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Normalized Root Mean Squared Error (NRMSE)
    - Visual plots available for comparison between imputation techniques
- Sustainability
    - Track the runtime, energy usage and CO2 emissions of each imputation method
- Forecasting
    - Predict future building energy consumption with either LightGBM or Linear Regression

## Getting Started
### 1. Clone the repository
