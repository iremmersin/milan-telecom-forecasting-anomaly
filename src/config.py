"""Project paths, dataset identifiers and zone definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

HOURLY_ALL_PARQUET = DATA_INTERIM / "milan_hourly_all_grids.parquet"
ZONES_CSV = DATA_PROCESSED / "milan_zones_hourly.csv"

KAGGLE_DATASET = "freckled/telecom"
GRID_DATASET = "marcodena/mobile-phone-activity"
GRID_FILE = "milano-grid.geojson"

TRAIN_END = "2013-12-08 23:00"
TEST_START, TEST_END = "2013-12-09 00:00", "2013-12-19 23:00"

STRESS_START = "2013-12-20 00:00"

# Local time (Europe/Rome), 62 days
START = "2013-11-01 00:00"
END = "2014-01-01 23:00"

# Inter vs Milan at San Siro
DERBY_DAY = "2013-12-22"

# (longitude, latitude), matching GeoJSON coordinate order
LANDMARKS = {
    "Duomo (Center)": (9.1900, 45.4642),
    "San Siro Stadium": (9.1240, 45.4781),
    "Bocconi University": (9.1890, 45.4497),
    "Milano Centrale": (9.2044, 45.4862),
}

# GridID = row * 100 + col + 1
ZONES = {
    "duomo": [4959, 4960, 4961, 5059, 5060, 5061, 5159, 5160, 5161],
    "san_siro": [5636, 5637, 5638, 5639, 5640, 5736, 5737, 5738, 5739, 5740,
             5836, 5837, 5838, 5839, 5840, 5936, 5937, 5938, 5939, 5940],
    "bocconi": [4460],
}
