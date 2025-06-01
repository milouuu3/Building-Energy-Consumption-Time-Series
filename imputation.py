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


def linear_regression(df_masked, df_true=None):
    df_imputed = df_masked.copy()

    for col in df_masked.columns:
        # Get the masked samples
        if df_true is not None:
            masked = df_masked[col].isna() & df_true[col].notna()
        else:
            masked = df_masked[col].isna()

        not_missing = df_masked[col].notna()

        if masked.sum() == 0:
            continue

        X = df_masked.drop(col, axis=1)
        y = df_masked[col]

        X_train = X.loc[not_missing].dropna()
        y_train = y.loc[X_train.index]
        X_pred = X.loc[masked].dropna()

        if X_train.empty or X_pred.empty:
            continue

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        df_imputed.loc[X_pred.index, col] = y_pred

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
    samples = rng.uniform(0, 1, size=df.shape) < missing
    df_masked = df.copy()
    df_masked.values[samples] = np.nan
    return df_masked, samples


def create_time_gap_data(df, bcount=3, bsize=24, seed=42):
    np.random.seed(seed)
    df_masked = df.copy()
    samples = np.zeros(df.shape, dtype=bool)

    for _ in range(bcount):
        # Select arbitrary starting index
        x = np.random.randint(0, len(df) - bsize + 1)
        y = x + bsize
        mask_col = np.random.choice(
            df.columns, size=np.random.randint(1, df.shape[1] + 1), replace=False
        )
        # Select subset of columns to mask
        cols = [df.columns.get_loc(i) for i in mask_col]
        df_masked.iloc[x:y, cols] = np.nan
        samples[x:y, cols] = True

    return df_masked, samples


def evaluate_imputation(df, df_masked, samples, method):
    if method == "Linear Regression":
        df_imputed = impute_data(df_masked, method, df)
    else:
        df_imputed = impute_data(df_masked, method)

    errors = {}
    for i, col in enumerate(df.columns):
        mask = samples[:, i]
        y_true = df.values[:, i][mask]
        y_pred = df_imputed.values[:, i][mask]

        valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.max(y_true) - np.min(y_true))

        errors[col] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse}
    return errors


def impute_data(df, method=None, df_true=None):
    if method == "LOCF":
        return locf(df)
    elif method == "NOCB":
        return nocb(df)
    elif method == "Linear Interpolation":
        return linear_interpolation(df)
    elif method == "Linear Regression":
        return linear_regression(df, df_true)
    elif method == "LightGBM":
        return lightgbm(df)


def mask_data(df, method=None):
    if method == "MCAR (Random Missingness)":
        return create_mcar_data(df)
    elif method == "Block Missingness (Time Gaps)":
        return create_time_gap_data(df)
