from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
import plotly.express as px
from imputation import create_mcar_data, evaluate_imputation, impute_data
from forecasting import forecast_data, plot_forecast
import base64
import io
import pandas as pd
import eco2ai
from codecarbon import EmissionsTracker
import time

methods = [
    "LOCF",
    "NOCB",
    "Linear Interpolation",
    "Linear Regression",
    "LightGBM",
]


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
            children=html.Div(html.A("Select CSV File")),
            multiple=False,
            accept=".csv",
            className="class-upload",
        ),
        # Store dataset
        dcc.Store(id="store-dataset"),
        dcc.Store(id="store-imputed-dataset"),
        dcc.Store(id="store-energy"),
        html.Div(id="output-data-upload"),
        # Create extra tabs for differen viewing levels
        dcc.Tabs(
            id="id-tabs",
            parent_className="class-tabs",
            className="tabs-container",
            children=[
                # Dataset level imputation
                dcc.Tab(
                    label="Dataset Imputation Evaluation",
                    children=[
                        html.Button("Run", id="button-run-dataset", n_clicks=0),
                        html.Div(id="dataset-results"),
                        # Download imputed data button
                        html.H2("Download Imputed Data"),
                        dcc.Dropdown(
                            id="dropdown-download-imputation-method",
                            placeholder="Select Imputation Method",
                            options=[{"label": m, "value": m} for m in methods],
                        ),
                        html.Button("Download Imputed Dataset", id="button-download"),
                        dcc.Download(id="download-imputed-data"),
                    ],
                    className="class-tab",
                    selected_className="tab--selected",
                ),
                # Computation Energy Consumption (per method)
                dcc.Tab(
                    label="Computational Energy Consumption",
                    children=html.Div(id="computational-energy-consumption"),
                    className="class-tab",
                    selected_className="tab--selected",
                ),
                # Forecasting building level
                dcc.Tab(
                    label="Forecasting Building Energy Consumption",
                    children=[
                        dcc.Dropdown(
                            id="dropdown-imputation-method",
                            placeholder="Select Imputation Method",
                            options=[{"label": i, "value": i} for i in methods],
                        ),
                        dcc.Dropdown(id="dropdown-building", placeholder="Select Building Column"),
                        dcc.Slider(1, 30, 1, value=7, id="slider-lags"),
                        html.Br(),
                        html.Button("Run forecast", id="button-run-forecast"),
                        dcc.Graph(id="plot-forecasting"),
                    ],
                    className="class-tab",
                    selected_className="tab--selected",
                ),
            ],
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
        return html.Div(html.P("There was an error processing this file.")), None

    return None, df


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
            return data, html.Div(html.H5(f"Succesfully uploaded: {names}"))
    return None, html.Div(html.H5("No file uploaded yet..."))


def evaluate(df, df_masked, samples):
    imputation_error = []
    for method in methods:
        print(f"Imputing with {method}")
        try:
            # First impute on masked data and get error
            error = evaluate_imputation(df.copy(), df_masked.copy(), samples, method)
            df_error = pd.DataFrame(error)
            df_error = df_error.reset_index()

            imputation_error.append(
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
            imputation_error.append(
                html.Div(
                    [
                        html.H4(f"Error metrics for {method}"),
                        html.P(f"Error: {e}"),
                    ],
                )
            )
    return imputation_error


def impute_original_data(df):
    imputed_data = {}
    energy = {}

    for method in methods:
        try:
            # Setup the tracking for emissions, consumption and runtime
            eco2ai_tracker = eco2ai.Tracker(
                project_name=f"eco2ai_{method}", file_name="eco2ai.csv"
            )
            codecarbon_tracker = EmissionsTracker(
                project_name=f"codecarbon_{method}", output_file="codecarbon.csv"
            )

            eco2ai_tracker.start()
            codecarbon_tracker.start()
            start = time.time()

            df_imputed = impute_data(df.copy(), method)

            end = time.time()
            codecarbon_tracker.stop()
            eco2ai_tracker.stop()

            df_eco2ai = pd.read_csv("eco2ai.csv")
            df_codecarbon = pd.read_csv("codecarbon.csv")

            codecarbon_measurements = df_codecarbon.iloc[-1]
            eco2ai_measurements = df_eco2ai.iloc[-1]

            elapsed_time = end - start
            imputed_data[method] = df_imputed.to_json(orient="split")
            energy[method] = {
                "Runtime (seconds)": elapsed_time,
                "CodeCarbon Emissions (kg)": codecarbon_measurements["emissions"],
                "CodeCarbon Energy (kWh)": codecarbon_measurements["energy_consumed"],
                "eco2AI Emissions (kg)": eco2ai_measurements["CO2_emissions(kg)"],
                "eco2AI Energy (kWh)": eco2ai_measurements["power_consumption(kWh)"],
            }
        except Exception as e:
            print(f"Error {methods}: {e}")
    return imputed_data, energy


@callback(
    Output("dataset-results", "children"),
    Output("store-imputed-dataset", "data"),
    Output("store-energy", "data"),
    Input("button-run-dataset", "n_clicks"),
    State("store-dataset", "data"),
)
def run_imputation(n_clicks, data):
    if n_clicks and data:
        df = pd.read_json(io.StringIO(data), orient="split")

        df_masked, samples = create_mcar_data(df.copy())
        imputation_error = evaluate(df, df_masked, samples)
        imputed_data, energy = impute_original_data(df)

        return imputation_error, imputed_data, energy

    return html.Div(html.H5("Click run to start evaluation")), None, None


@callback(
    Output("dropdown-building", "options"),
    Input("store-imputed-dataset", "data"),
    Input("dropdown-imputation-method", "value"),
)
def get_data(imputed_data, method):
    if not imputed_data or not method:
        return []
    data = imputed_data.get(method)
    if not data:
        return []
    df_imputed = pd.read_json(io.StringIO(data), orient="split")
    return [{"label": i, "value": i} for i in df_imputed.columns]


@callback(
    Output("plot-forecasting", "figure"),
    Input("button-run-forecast", "n_clicks"),
    State("store-imputed-dataset", "data"),
    State("dropdown-building", "value"),
    State("slider-lags", "value"),
    State("dropdown-imputation-method", "value"),
)
def run_forecast(n_clicks, imputed_data, col, n_lags, method):
    if n_clicks and imputed_data and method:
        df = imputed_data.get(method)
        if df is None:
            print(f"Forecasting: imputed data not found of method: {method}")
            return None

        df_imputed = pd.read_json(io.StringIO(df), orient="split")
        y_test, y_pred = forecast_data(df_imputed, col, n_lags)
        return plot_forecast(y_test, y_pred)
    return None


@callback(Output("computational-energy-consumption", "children"), Input("store-energy", "data"))
def visualize_energy_consumption(data):
    if not data:
        return html.Div(html.H5("No measurements available yet..."))

    # Melt the measurements columns from CodeCarbon and eco2AI together per metric
    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "Method"}, inplace=True)
    df_co2 = df.melt(
        id_vars="Method",
        value_vars=["CodeCarbon Emissions (kg)", "eco2AI Emissions (kg)"],
        value_name="Emissions (kg)",
    )
    df_consump = df.melt(
        id_vars="Method",
        value_vars=["CodeCarbon Energy (kWh)", "eco2AI Energy (kWh)"],
        value_name="Energy Consumption (kWh)",
    )

    # Create bar figures
    fig_time = px.bar(df, x="Method", y="Runtime (seconds)", title="Runtime per method")
    fig_co2 = px.bar(
        df_co2, x="Method", y="Emissions (kg)", title="CO2 Emissions per method and tracker"
    )
    fig_consump = px.bar(
        df_consump,
        x="Method",
        y="Energy Consumption (kWh)",
        title="Energy Consumption per method and tracker",
    )

    return html.Div(
        [
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
            ),
            html.Div(
                [
                    dcc.Graph(figure=fig_time),
                    dcc.Graph(figure=fig_co2),
                    dcc.Graph(figure=fig_consump),
                ]
            ),
        ]
    )


@callback(
    Output("download-imputed-data", "data"),
    Input("button-download", "n_clicks"),
    State("store-imputed-dataset", "data"),
    State("dropdown-download-imputation-method", "value"),
    prevent_initial_call=True,
)
def download_data(n_clicks, data, method):
    if n_clicks and data and method:
        data = data.get(method)
        if not data:
            return None
        df = pd.read_json(io.StringIO(data), orient="split")
        return dcc.send_data_frame(df.to_csv, "imputed-data.csv")
    return None


if __name__ == "__main__":
    app.run(debug=True)
