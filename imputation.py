from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def locf(df):
    return df.ffill()


def nocb(df):
    return df.bfill()


def linear_interpolation(df):
    return df.interpolate(method="linear", limit_direction="both")


def linear_regression(df):
    df_imputed = df.copy()

    for col in df.columns:
        missing = df[col].isna()
        not_missing = df[col].notna()

        if missing.sum() == 0:
            continue

        X = df.drop(col, axis=1)
        y = df[col]

        X_train = X.loc[not_missing]
        y_train = y.loc[not_missing]
        X_pred = X.loc[missing]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        df_imputed.loc[missing, col] = y_pred

    return df_imputed


def lightgbm(df):
    df_imputed = df.copy()

    for col in df.columns:
        missing = df[col].isna()
        not_missing = df[col].notna()

        if missing.sum() == 0:
            continue

        X = df.drop(col, axis=1)
        y = df[col]

        X_train = X.loc[not_missing]
        y_train = y.loc[not_missing]
        X_pred = X.loc[missing]

        model = LGBMRegressor(force_col_wise=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        df_imputed.loc[missing, col] = y_pred

    return df_imputed


def create_mcar_data(df, missing=0.2, seed=42):
    rng = np.random.default_rng(seed)
    df_masked = df.copy()
    samples = rng.uniform(0, 1, size=df.shape)
    df_masked.values[samples < missing] = np.nan
    return df_masked


def evaluate_imputation(df, method):
    masked = df.isna()
    df_imputed = impute_data(df, method)

    errors = {}
    for col in df.columns:
        y_pred = df_imputed.loc[masked[col], col]
        y_true = df.loc[masked[col], col]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.maximum(y_true) - np.minimum(y_true))

        errors[col] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse}
    return errors


def impute_data(df, method=None):
    if method == "LOCF":
        return locf(df)
    elif method == "NOCB":
        return nocb(df)
    elif method == "Linear Interpolation":
        return linear_interpolation(df)
    elif method == "Linear Regression":
        return linear_regression(df)
    elif method == "LightGBM":
        return lightgbm(df)
