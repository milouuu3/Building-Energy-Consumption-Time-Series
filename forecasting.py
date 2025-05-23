from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor


def create_lags(df, col, n_lags=7):
    df_lags = df.copy()
    for i in range(1, n_lags + 1):
        df_lags[f"Lag_{i}"] = df[col].shift(i)
    return df_lags


def forecast_data(df, col, model_type, n_lags=7, train_size=0.8):
    df_lags = create_lags(df, col, n_lags=n_lags)

    X = df_lags.drop(columns=[col])
    y = df_lags[col]

    p = int(len(X) * train_size)
    X_train, X_test = X.iloc[:p], X.iloc[p:]
    y_train, y_test = y.iloc[:p], y.iloc[p:]

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "LightGBM":
        model = LGBMRegressor(force_col_wise=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred
