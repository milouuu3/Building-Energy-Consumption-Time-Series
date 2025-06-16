# Building-Energy-Consumption-Time-Series
This repository contains the experiment notebooks and framework for my bachelor thesis about Imputation on Building Energy Time Series Time Series data.

# Experiments
- Masked by MCAR (20%)
- Run on Linux

## Imputation Techniques
    - Last Observation Carried Forward (LOCF)
    - Next Observation Carried Backwards (NOCB)
    - Linear Interpolation
    - Multiple Imputation by Chained Equation (MICE)
    - Linear Regression
    - K-Nearest Neighbor (KNN)
    - LightGBM

## Datasets
- https://www.kaggle.com/datasets/mexwell/household-load-and-solar-generation
- https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set/data
- https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2/data

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
    - Predict future building energy consumption with either a LightGBM or Linear Regression model

# Getting Started
## 0.1 Overview of necessary Python libraries
    - CodeCarbon
    - Dash
    - Jupyter Notebook
    - LightGBM
    - Matplotlib
    - Numpy
    - Pandas
    - Scikit-Learn

## 0.2 Open your terminal
Open your terminal:
- **Linux** (recommended): `Ctrl+Alt+T`
- **macOS**: `Cmd + Space`, then type `terminal`, and press `Enter`
- **Windows**: `Win + R`, then type `cmd` or `powershell`, and press `Enter`

## 1 Clone the repository
`git clone git@github.com:milouuu3/Building-Energy-Consumption-Time-Series.git`

## 2. Navigate to this folder
`cd Building-Energy-Consumption-Time-Series`

## 3.1 Run the 'setup' bash file to install all dependencies
Type `./setup.sh` in your terminal

## 3.2 Run the 'run' bash file to start the application
Type `./run.sh` in your terminal

## 4 To quit the application
Hold `Ctrl + C`
