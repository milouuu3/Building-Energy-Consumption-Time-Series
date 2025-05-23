from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
from imputation import (
    impute_data,
    create_mcar_data,
    evaluate_imputation,
    forcast_data,
)
import base64
import io
import pandas as pd

app = Dash()

app.layout = html.Div(
    [
        html.H1("Building Energy Consumption Time Series"),
        # Dataset
        html.H2(children="Upload Building Energy Consumption Dataset"),
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
        dcc.Loading(
            id="data-upload-loading",
            type="circle",
            children=html.Div(id="output-data-upload"),
        ),
    ],
    style={"padding": "20px"},
)


def preprocess_data(df):
    # Find and set the timestamp column as index
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
            except Exception:
                pass
    # Filter to only numeric columns
    df = df.select_dtypes(include="number")
    return df.sort_index()


def parse_contents(contents, filename):
    if not filename.endswith(".csv"):
        return html.Div(["Only CSV files are supported."]), None

    ctype, cstring = contents.split(",")
    decoded = base64.b64decode(cstring)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df = preprocess_data(df)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), None

    return df


methods = [
    "LOCF",
    "NOCB",
    "Linear Interpolation",
    "Linear Regression",
    "LightGBM",
]


def create_datatable(df, method):
    df = df.head().reset_index()
    return html.Div(
        [
            html.H3(f"Imputed with {method}"),
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
            ),
        ],
    )


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(content, names):
    if content is not None:
        print("Dataset is uploaded...")
        children = []
        for c, n in zip(content, names):
            print(f"Processing dataset: {n}")
            df = parse_contents(c, n)

            if df is None:
                print("Read error in file...")
                continue

            for method in methods:
                print(f"Running imputation: {method}")
                try:
                    df_imputed = impute_data(df.copy(), method)
                    missing = df_imputed.isna().sum().sum()
                    if missing > 0:
                        children.append(
                            html.Div(
                                [
                                    html.P(
                                        f"Warning: still contains {missing} missing values. Forecasting accuracy might be affected.",
                                        style={
                                            "color": "white",
                                            "padding": "20px",
                                            "marginBottom": "15px",
                                            "backgroundColor": "#f44336",
                                        },
                                    ),
                                ]
                            )
                        )
                    children.append(create_datatable(df_imputed, method))
                except Exception as e:
                    children.append(
                        html.Div(
                            [
                                html.H3(f"Imputed with {method}"),
                                html.P(
                                    f"Error: {e}",
                                    style={
                                        "color": "white",
                                        "padding": "20px",
                                        "marginBottom": "15px",
                                        "backgroundColor": "#f44336",
                                    },
                                ),
                            ]
                        )
                    )
        print("Everything is processed...")
        return children


if __name__ == "__main__":
    app.run(debug=True)
