from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
import base64
import datetime
import io
import pandas as pd
from sklearn.impute import KNNImputer

app = Dash()

app.layout = html.Div(
    [
        # Dataset
        html.H2(
            children="Upload Building Energy Consumption Dataset", style={"textAlign": "center"}
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.Div(id="output-data-upload"),
    ],
    style={"padding": "20px"},
)


def parse_contents(contents, filename, date):
    if not filename.endswith(".csv"):
        return html.Div(["Only CSV files are supported."]), None

    ctype, cstring = contents.split(",")
    decoded = base64.b64decode(cstring)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), None

    return (
        html.Div(
            [
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),
                dash_table.DataTable(
                    df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]
                ),
            ],
        ),
        df,
    )


def format_imputed_result(df_imputed, method):
    return html.Div(
        [
            html.H4(f"Imputed with {method}"),
            dash_table.DataTable(
                df_imputed.round(2).to_dict("records"),
                [{"name": i, "id": i} for i in df_imputed.columns],
                style_table={"overflowX": "auto"},
                page_size=10,
            ),
        ],
        style={"marginBottom": "30px"},
    )


methods = ["LOCF", "NOCB", "Linear Interpolation"]


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_output(content, names, dates):
    if content is not None:
        children = []
        for c, n, d in zip(content, names, dates):
            div, df = parse_contents(c, n, d)
            children.append(div)

            if df is None:
                continue

            for method in methods:
                df_imputed = impute_data(df.copy(), method)
                children.append(format_imputed_result(df_imputed, method))
        return children


def locf(df):
    return df.ffill()


def nocb(df):
    return df.bfill()


def linear_interpolation(df):
    return df.interpolate(method="linear")


def linear_regression(df):
    pass


def lightgbm(df):
    pass


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


if __name__ == "__main__":
    app.run(debug=True)
