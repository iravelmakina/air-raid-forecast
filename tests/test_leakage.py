"""Leakage and label checks (implementation plan, 'Automated checks')."""
import pandas as pd

from src.data import TZ, REGION, build_night_labels
from src.evaluate import history_at, walk_forward
from src.features import compute_features


def kyiv(ts: str) -> pd.Timestamp:
    """Kyiv local time string -> UTC timestamp (as stored in the dataset)."""
    return pd.Timestamp(ts, tz=TZ).tz_convert("UTC")


def alerts_df(rows):
    return pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"])


NIGHT = pd.Timestamp("2026-01-10")


def labels_up_to(night, value=0, n=40):
    idx = pd.date_range(night - pd.Timedelta(days=n), night, freq="D")
    return pd.Series(value, index=idx)


def test_1_alert_crossing_22_labels_night():
    alerts = alerts_df([
        (REGION, kyiv("2026-01-09 12:00"), kyiv("2026-01-09 12:20"), False),
        (REGION, kyiv("2026-01-10 21:50"), kyiv("2026-01-10 22:10"), False),
        ("Lvivska oblast", kyiv("2026-01-12 12:00"), kyiv("2026-01-12 12:30"), False),  # moves the snapshot past the night
    ])
    labels = build_night_labels(alerts)
    assert labels[NIGHT] == 1
    assert labels[NIGHT - pd.Timedelta(days=1)] == 0


def test_2_alert_starting_after_cutoff_is_invisible():
    alerts = alerts_df([
        (REGION, kyiv("2026-01-10 18:01"), kyiv("2026-01-10 18:30"), False),
        ("Lvivska oblast", kyiv("2026-01-10 18:05"), kyiv("2026-01-10 18:40"), False),
    ])
    h = history_at(NIGHT, labels_up_to(NIGHT), alerts)
    assert h.alerts.empty
    f = compute_features(h)
    assert f["kyiv_daytime_alert"] == 0
    assert f["other_regions_daytime"] == 0
    assert f["approach_regions_afternoon"] == 0


def test_3_running_alert_has_end_erased():
    alerts = alerts_df([(REGION, kyiv("2026-01-10 17:30"), kyiv("2026-01-10 19:00"), False)])
    h = history_at(NIGHT, labels_up_to(NIGHT), alerts)
    assert h.alerts["finished_at"].isna().all()


def test_4_walk_forward_never_sees_predicted_night():
    labels = labels_up_to(NIGHT, value=1)
    alerts = alerts_df([(REGION, kyiv("2026-01-10 23:00"), kyiv("2026-01-10 23:30"), False)])
    seen = []

    def spy(history):
        night = history.cutoff.tz_convert(TZ).normalize().tz_localize(None)
        assert history.labels.index.max() < night          # no label of tonight or later
        assert (history.alerts["started_at"] < history.cutoff).all()
        seen.append(night)
        return 0.5

    nights = labels.index[-5:]
    walk_forward(spy, nights, labels, alerts)
    assert len(seen) == 5


def test_5_night_too_close_to_snapshot_is_excluded():
    alerts = alerts_df([
        (REGION, kyiv("2026-01-09 12:00"), kyiv("2026-01-09 12:20"), False),
        (REGION, kyiv("2026-01-10 23:00"), kyiv("2026-01-10 23:30"), False),
        ("Lvivska oblast", kyiv("2026-01-11 20:00"), kyiv("2026-01-11 20:30"), False),  # snapshot only 14 h after the night ended
    ])
    labels = build_night_labels(alerts)
    assert NIGHT not in labels.index
