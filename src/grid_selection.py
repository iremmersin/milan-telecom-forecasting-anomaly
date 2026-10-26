# Grid cell selection and verification.

# 1. Cells with the largest traffic spike on derby night vs. other Sunday evenings.
# 2. Highest-traffic cells overall (city-center candidates).
# 3. Landmark coordinates mapped to grid cells via the GeoJSON grid.


import json
import numpy as np
import pandas as pd
import kagglehub

from config import (
    HOURLY_ALL_PARQUET, GRID_DATASET, GRID_FILE, DERBY_DAY, LANDMARKS,
)

CHECK_CELLS = [5638, 5738, 5060, 5161]  # stadium and city-center reference cells
EXTRA_CELLS = [8169]                    # unexplained spike, located via GeoJSON


def derby_tables(df, derby_day):
    df = df.copy()
    df["date"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.day_of_week  # 6 = Sunday

    derby_date = pd.Timestamp(derby_day).date()
    evening = df[df["hour"].between(19, 23)]

    # Baseline: median of other Sunday evenings (robust to other match days)
    base = (
        evening[(evening["dow"] == 6) & (evening["date"] != derby_date)]
        .groupby("GridID")["internet"].median()
    )
    peak = evening[evening["date"] == derby_date].groupby("GridID")["internet"].max()
    return base, peak


def print_derby_report(base, peak):
    print("Sunday evening baseline distribution")
    print(base.describe())

    raw_ratio = (peak / base).replace([np.inf, -np.inf], np.nan).dropna()

    # Ratios inflate on tiny baselines, so drop the quietest 10% of cells
    ratio = raw_ratio[base.reindex(raw_ratio.index) > base.quantile(0.1)]
    top = ratio.nlargest(10)
    top10 = pd.DataFrame({
        "spike_ratio": top,
        "sunday_median": base.reindex(top.index),
        "derby_peak": peak.reindex(top.index),
    })
    print("\nTop 10 cells by spike ratio")
    print(top10.round(2))

    print("\nTop 10 cells by absolute increase (peak - baseline)")
    print((peak - base).dropna().nlargest(10).round(1))

    chk = pd.DataFrame({
        "base": base.reindex(CHECK_CELLS),
        "peak": peak.reindex(CHECK_CELLS),
    })
    chk["ratio"] = chk["peak"] / chk["base"]
    chk["increase"] = chk["peak"] - chk["base"]
    chk["rank_by_ratio"] = raw_ratio.rank(ascending=False).reindex(CHECK_CELLS)
    print("\nReference cells (unfiltered ranking)")
    print(chk.round(2))


def get_id(feature):
    props = feature["properties"]
    for key in ("cellId", "id"):
        if props.get(key) is not None:
            return int(props[key])
    return int(feature["id"])


def bbox(feature):
    coords = feature["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)


def print_geojson_report(grid):
    print("\nLandmark to grid cell mapping")
    for name, (lon, lat) in LANDMARKS.items():
        for feature in grid["features"]:
            lon_min, lon_max, lat_min, lat_max = bbox(feature)
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                cid = get_id(feature)
                row_idx, col_idx = divmod(cid - 1, 100)
                print(f"{name:20s} -> GridID: {cid:5d} (row {row_idx:2d}, col {col_idx:2d})")
                break
        else:
            print(f"{name:20s} -> no matching cell (check coordinate order)")

    print("\nCell centers (lat, lon)")
    for target in EXTRA_CELLS:
        feat = next((f for f in grid["features"] if get_id(f) == target), None)
        if feat is None:
            print(f"GridID {target}: not in GeoJSON")
            continue
        lon_min, lon_max, lat_min, lat_max = bbox(feat)
        print(f"GridID {target}: {(lat_min + lat_max) / 2:.5f}, {(lon_min + lon_max) / 2:.5f}")


def main():
    df = pd.read_parquet(HOURLY_ALL_PARQUET)

    base, peak = derby_tables(df, DERBY_DAY)
    print_derby_report(base, peak)

    print("\nTop 5 cells by total traffic")
    print(df.groupby("GridID")["internet"].sum().nlargest(5).round(1))

    try:
        geojson_path = kagglehub.dataset_download(GRID_DATASET, path=GRID_FILE)
        with open(geojson_path, "r", encoding="utf-8") as f:
            grid = json.load(f)
        print_geojson_report(grid)
    except Exception as e:
        print("\nGeoJSON check failed:", e)


if __name__ == "__main__":
    main()