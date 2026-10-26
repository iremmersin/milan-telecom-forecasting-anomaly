#Aggregate the raw Telecom Italia CSVs into hourly internet traffic per grid cell.

import sys
from pathlib import Path

import pyarrow # (fail early if the parquet engine is missing)
import pandas as pd
import kagglehub

from config import KAGGLE_DATASET, HOURLY_ALL_PARQUET

CHUNKSIZE = 5_000_000
MS_PER_HOUR = 3_600_000
RAW_FILES = ["data1.csv", "data2.csv"]


def build_hourly_all_grids(max_chunks_per_file=None):
    raw_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    parts = []
    for name in RAW_FILES:
        reader = pd.read_csv(
            raw_dir / name,
            usecols=["GridID", "TimeInterval", "internet"],
            chunksize=CHUNKSIZE,
        )
        for i, chunk in enumerate(reader):
            print(name, "chunk", i, flush=True)
            # One row per country code; NaN means no record
            chunk = chunk.dropna(subset=["internet"])
            # UTC offset is a whole number of hours, so flooring in UTC
            # is equivalent to flooring in local time
            chunk["hour"] = chunk["TimeInterval"] // MS_PER_HOUR
            parts.append(
                chunk.groupby(["GridID", "hour"], as_index=False)["internet"].sum()
            )
            if max_chunks_per_file is not None and i + 1 >= max_chunks_per_file:
                break
        print(name, "done", flush=True)

    # Merge groups split across chunk boundaries
    hourly = (
        pd.concat(parts)
        .groupby(["GridID", "hour"], as_index=False)["internet"]
        .sum()
    )
    hourly["ts"] = (
        pd.to_datetime(hourly["hour"] * MS_PER_HOUR, unit="ms", utc=True)
        .dt.tz_convert("Europe/Rome")
        .dt.tz_localize(None)
    )
    return hourly.drop(columns="hour")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    hourly = build_hourly_all_grids(max_chunks_per_file=2 if test_mode else None)

    HOURLY_ALL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_ALL_PARQUET)
    print(hourly.shape, hourly["ts"].min(), hourly["ts"].max())
    print("Saved:", HOURLY_ALL_PARQUET)