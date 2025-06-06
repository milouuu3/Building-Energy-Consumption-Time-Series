import numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

        Xf = X.fillna(X.mean())

        X_train = Xf.loc[not_missing]
        y_train = y.loc[not_missing]
        X_pred = Xf.loc[missing]

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
    samples = rng.uniform(0, 1, size=df.shape) < missing
    df_masked = df.copy()
    df_masked.values[samples] = np.nan
    return df_masked, samples


def create_timegap_data(df, gap_size=4, n=1, missing=0.2, seed=42):
    df_masked = df.copy()
    rng = np.random.default_rng(seed)
    threshold = np.maximum(1, int(df.shape[1] * missing))

    for _ in range(n):
        x = rng.integers(0, np.maximum(df.shape[0] - gap_size, 1))
        mask = rng.choice(df.columns, threshold, replace=False)
        df_masked.loc[df.index[x : x + gap_size], mask] = np.nan

    samples = df_masked.isna().values
    return df_masked, samples


def create_interval_data(df, interval=24):
    df_masked = df.copy()
    df_masked.iloc[::interval, :] = np.nan
    samples = df_masked.isna().values
    return df_masked, samples


def evaluate_imputation(df, df_masked, samples, method):
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


def mask_data(df, method):
    if method == "Missing Completely at Random":
        return create_mcar_data(df)
    elif method == "Time Gap Masking":
        return create_timegap_data(df)
    else:
        return create_interval_data(df)
