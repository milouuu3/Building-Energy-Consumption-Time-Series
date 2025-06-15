import plotly.graph_objs as go
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression


def create_lags(df, col, n_lags=7):
    df_lags = df.copy()
    for i in range(1, n_lags + 1):
        df_lags[f"Lag_{i}"] = df[col].shift(i)
    return df_lags


def split_data(X, y, train_size=0.8):
    p = int(len(X) * train_size)
    X_train, X_test = X.iloc[:p], X.iloc[p:]
    y_train, y_test = y.iloc[:p], y.iloc[p:]
    return X_train, X_test, y_train, y_test


def forecast_data(df, target, input_features, method, n_lags=7, train_size=0.8):
    df_lags = create_lags(df, target, n_lags=n_lags)

    X = df_lags[input_features]
    y = df_lags[target]

    X_train, X_test, y_train, y_test = split_data(X, y, train_size=train_size)

    if method == "Linear Regression":
        model = LinearRegression()
    else:
        model = LGBMRegressor(force_col_wise=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred


def plot_forecast(y_test, y_pred):
    fig = go.Figure()

    # Predictions as scatter
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            name="y_test",
            line=dict(color="blue"),
            mode="markers",
        )
    )

    # Perfect line
    fig.add_trace(
        go.Scatter(
            x=[np.min(y_test), np.max(y_test)],
            y=[np.min(y_test), np.max(y_test)],
            name="y_pred",
            line=dict(color="red"),
            mode="lines",
        )
    )

    fig.update_layout(
        title="Predicted vs Actual",
        showlegend=True,
        xaxis_title="Actual values",
        yaxis_title="Predicted Values",
    )
    return fig
