"""Load alert data and build the night-level target.

Target: for the night that STARTS on date d, was Kyiv City under an air raid
alert at any moment between d 22:00 and d+1 06:00 (Kyiv local time)? 1 = yes, 0 = no.
"""
from pathlib import Path
import urllib.request

import pandas as pd

TZ = "Europe/Kyiv"
REGION = "Kyiv City"
NIGHT_START_H = 22   # 22:00 on day d
NIGHT_END_H = 30     # 06:00 on day d+1  (= 24 + 6)
FORECAST_H = 18      # forecast is made at 18:00 on day d

# Pinned dataset version: the source repo updates daily, so we fix one commit
# to make every result reproducible.
DATA_COMMIT = "7be33582f840bf99b14f92118b40f6131b492e7c"
DATA_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    f"{DATA_COMMIT}/datasets/volunteer_data_en.csv"
)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "volunteer_data_en.csv"


def load_alerts() -> pd.DataFrame:
    """All alerts (every region), times in UTC. Downloads the pinned file once."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["started_at", "finished_at"])
    return df.sort_values("started_at").reset_index(drop=True)


def night_window(night: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of the night that starts on date `night`, in UTC."""
    day = pd.Timestamp(night.date()).tz_localize(TZ)
    start = day + pd.Timedelta(hours=NIGHT_START_H)
    end = day + pd.Timedelta(hours=NIGHT_END_H)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def forecast_time(night: pd.Timestamp) -> pd.Timestamp:
    """The cutoff: 18:00 Kyiv time on the evening the night starts, in UTC."""
    day = pd.Timestamp(night.date()).tz_localize(TZ)
    return (day + pd.Timedelta(hours=FORECAST_H)).tz_convert("UTC")


def build_night_labels(alerts: pd.DataFrame) -> pd.Series:
    """One row per night: 1 if any Kyiv City alert overlaps the night window.

    Only nights that had fully ended before the data snapshot are included,
    otherwise the last night could be labelled 'quiet' just because it was not over yet.
    """
    kyiv = alerts[alerts["region"] == REGION]
    snapshot_time = alerts["started_at"].max()  # proxy for when the file was built

    first = kyiv["started_at"].min().tz_convert(TZ).normalize().tz_localize(None)
    last = snapshot_time.tz_convert(TZ).normalize().tz_localize(None)
    nights = pd.date_range(first, last, freq="D")

    labels = {}
    for night in nights:
        start, end = night_window(night)
        if end > snapshot_time:
            continue  # night not finished in this data snapshot
        overlaps = (kyiv["started_at"] < end) & (kyiv["finished_at"] > start)
        labels[night] = int(overlaps.any())
    return pd.Series(labels, name="alert_night").rename_axis("night")
