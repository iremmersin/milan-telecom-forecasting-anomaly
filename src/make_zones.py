# Build hourly internet traffic per zone.

import pandas as pd
from config import HOURLY_ALL_PARQUET, ZONES_CSV, ZONES, START, END


def build_zones(df):
    full_idx = pd.date_range(START, END, freq="h")
    series = {}
    for zone, cells in ZONES.items():
        s = df[df["GridID"].isin(cells)].groupby("ts")["internet"].sum()
        s = s.reindex(full_idx)  # hours without records stay NaN
        print(f"{zone:10s} cells: {len(cells):2d} | missing hours: {int(s.isna().sum())}")
        series[zone] = s
    return pd.DataFrame(series).rename_axis("ts")


if __name__ == "__main__":
    df = pd.read_parquet(HOURLY_ALL_PARQUET)
    out = build_zones(df)

    ZONES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ZONES_CSV)
    print(out.shape, "->", ZONES_CSV)