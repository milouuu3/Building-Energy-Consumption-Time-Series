import pandas as pd

df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    parse_dates={"timestamp": ["Date", "Time"]},
    na_values="?",
    infer_datetime_format=True,
    low_memory=False,
)

df.set_index("timestamp", inplace=True)
df_resampled = df.resample("60T").mean()
df_resampled.to_csv("uci_60min.csv")
