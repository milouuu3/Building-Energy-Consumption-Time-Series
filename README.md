# Building-Energy-Consumption-Time-Series
This repository contains the experiment notebooks and custom built-in framework for my bachelor thesis about imputation on building energy time series data.

# Experiments
- Masked by MCAR (20%)
- Experiments were run on Ubuntu 24.04.2 LTS
- Used Error metrics: MAE and NRMSE

## Used Imputation Techniques
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
- Available Imputation Techniques
    - Last Observation Carried Forward (LOCF)
    - Next Observation Carried Backwards (NOCB)
    - Linear Interpolation
    - Linear Regression
    - LightGBM
- Evaluation
    - Different Masking Techniques
        - Missing completely at random (MCAR)
        - Time Gap masking (Continuous Block)
        - Fixed Interval masking (Regular Intervals)
    - Error metrics available
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Normalized Root Mean Squared Error (NRMSE)
    - Visual plots available for comparison between imputation techniques
- Imputed data can be downloaded for downstream use
- Sustainability
    - Track the runtime, energy usage and CO2 emissions of each imputation method
- Forecasting
    - Predict future building energy consumption with either a LightGBM or linear regression model

# Getting Started
## 0.1. Overview of necessary Python libraries
    - CodeCarbon
    - Dash
    - Jupyter Notebook
    - LightGBM
    - Matplotlib
    - Numpy
    - Pandas
    - Scikit-Learn

## 0.2. Open your terminal
Open your terminal:
- **Linux** (recommended): `Ctrl+Alt+T`
- **macOS**: `Cmd + Space`, then type `terminal`, and press `Enter`

## 1. Clone the repository
`git clone git@github.com:milouuu3/Building-Energy-Consumption-Time-Series.git`

## 2. Navigate to this folder
`cd Building-Energy-Consumption-Time-Series`

## 3.1. Run the 'setup' bash file to install all dependencies
Type `./setup.sh` in your terminal

## 3.2. Run the 'run' bash file to start the application
Type `./run.sh` in your terminal

## 4. Go to this webpage
By default, the app runs at: [`http://127.0.0.1:8050/`](http://127.0.0.1:8050/) (check your terminal for the exact address)

## 5 To. quit the application
Hold `Ctrl + C`
