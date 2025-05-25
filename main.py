from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
from imputation import create_mcar_data, evaluate_imputation, impute_data
from forecasting import forecast_data, plot_forecast
import base64
import io
import pandas as pd

app = Dash()

app.layout = html.Div(
    [
        # Header
        html.Div(
            className="app-header",
            children=[
                html.H1("Building Energy Consumption Time Series", className="app-header--title")
            ],
        ),
        # Dataset upload
        html.H2(children="Upload Building Energy Consumption Dataset"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select CSV File")]),
            style={
                "backgroundColor": "#ffffff",
                "border": "1.8px solid #2C3E50",
                "borderRadius": "8px",
                "padding": "40px 20px",
                "margin": "20px auto",
                "maxWidth": "600px",
                "cursor": "pointer",
                "textAlign": "center",
            },
            multiple=False,
            accept=".csv",
        ),
        # Store dataset
        dcc.Store(id="store-dataset"),
        dcc.Store(id="store-imputed-dataset"),
        html.Div(id="output-data-upload"),
        # Create extra tabs for differen viewing levels
        dcc.Loading(
            id="data-upload-loading",
            type="circle",
            children=dcc.Tabs(
                id="tabs",
                children=[
                    # Dataset level
                    dcc.Tab(
                        label="dataset-evaluation",
                        children=[
                            html.Button("Run", id="button-run-dataset", n_clicks=0),
                            html.Div(id="dataset-results"),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #bdc3c7",
                            "borderBottom": "none",
                            "borderTopLeftRadius": "6px",
                            "borderTopRightRadius": "6px",
                            "padding": "12px 24px",
                            "fontFamily": "sans-serif",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#2C3E50",
                            "cursor": "pointer",
                            "marginRight": "5px",
                        },
                        selected_style={
                            "backgroundColor": "#CDD0D3",
                            "border": "1px solid #ffffff",
                            "padding": "12px 24px",
                            "fontWeight": "600",
                            "borderTopLeftRadius": "6px",
                            "borderTopRightRadius": "6px",
                        },
                    ),
                    # Building level
                    dcc.Tab(
                        label="building-forecasting",
                        children=[
                            dcc.Dropdown(
                                id="dropdown-building", placeholder="Select Building Column"
                            ),
                            dcc.Slider(1, 30, 1, value=7, id="slider-lags"),
                            html.Br(),
                            html.Button("Run forecast", id="button-run-forecast"),
                            dcc.Graph(id="plot-forecasting"),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #bdc3c7",
                            "borderBottom": "none",
                            "borderTopLeftRadius": "6px",
                            "borderTopRightRadius": "6px",
                            "padding": "12px 24px",
                            "fontFamily": "sans-serif",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#2C3E50",
                            "cursor": "pointer",
                            "marginRight": "5px",
                        },
                        selected_style={
                            "backgroundColor": "#CDD0D3",
                            "border": "1px solid #ffffff",
                            "padding": "12px 24px",
                            "fontWeight": "600",
                            "borderTopLeftRadius": "6px",
                            "borderTopRightRadius": "6px",
                        },
                    ),
                ],
            ),
        ),
    ],
)


def preprocess_data(df: pd.DataFrame):
    # Find and set the timestamp column as index if possible
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
            except Exception as e:
                print(e)
    # Filter only numeric columns
    df = df.select_dtypes(include="number")
    return df.sort_index()


def parse_contents(contents, filename):
    _, cstring = contents.split(",")
    decoded = base64.b64decode(cstring)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df = preprocess_data(df)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), None

    return None, df


methods = [
    "LOCF",
    "NOCB",
    "Linear Interpolation",
    "Linear Regression",
    "LightGBM",
]


@callback(
    Output("store-dataset", "data"),
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(content, names):
    if content is not None:
        div, df = parse_contents(content, names)
        if div:
            return None, div
        else:
            data = df.to_json(orient="split")
            return data, html.Div([f"Succesfully uploaded: {names}"])
    return None, html.Div(["No file uploaded yet..."])


@callback(
    Output("dataset-results", "children"),
    Output("store-imputed-dataset", "data"),
    Input("button-run-dataset", "n_clicks"),
    State("store-dataset", "data"),
)
def run_imputation(n_clicks, data):
    if n_clicks and data:
        df = pd.read_json(io.StringIO(data), orient="split")

        df_masked, samples = create_mcar_data(df.copy())

        results = []
        for method in methods:
            print(f"Imputing with {method}")
            try:
                error = evaluate_imputation(df.copy(), df_masked.copy(), samples, method)
                df_error = pd.DataFrame(error)
                df_error = df_error.reset_index()
                results.append(
                    html.Div(
                        [
                            html.H4(f"Error metrics for {method}"),
                            dash_table.DataTable(
                                data=df_error.to_dict("records"),
                                columns=[{"name": i, "id": i} for i in df_error.columns],
                            ),
                        ]
                    )
                )
            except Exception as e:
                results.append(
                    html.Div(
                        [
                            html.H3(f"Error metrics for {method}"),
                            html.P(f"Error: {e}"),
                        ],
                    )
                )
        # Impute the original data using the chosen method without mask
        method_choice = "LightGBM"
        print(f"Method {method_choice} is chosen as final method")
        df_imputed = impute_data(df.copy(), method_choice)
        imputed_data = df_imputed.to_json(orient="split")
        return results, imputed_data

    return html.Div(["Click run to start evaluation"]), None


@callback(Output("dropdown-building", "options"), Input("store-imputed-dataset", "data"))
def update_options(imputed_data):
    if imputed_data is None:
        return []
    df_imputed = pd.read_json(io.StringIO(imputed_data), orient="split")
    return [{"label": i, "value": i} for i in df_imputed.columns]


@callback(
    Output("plot-forecasting", "figure"),
    Input("button-run-forecast", "n_clicks"),
    State("store-imputed-dataset", "data"),
    State("dropdown-building", "value"),
    State("slider-lags", "value"),
)
def run_forecast(n_clicks, imputed_data, col, n_lags):
    if n_clicks and imputed_data:
        df_imputed = pd.read_json(io.StringIO(imputed_data), orient="split")
        y_test, y_pred = forecast_data(df_imputed, col, n_lags)
        fig = plot_forecast(y_test, y_pred)
        return fig


if __name__ == "__main__":
    app.run(debug=True)
