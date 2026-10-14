"""Four fixed error-analysis checks (ADR D15). Read saved test outputs only;
findings go to the README and never trigger changes or re-runs."""
import pandas as pd

CAL_BINS = [0.0, 0.5, 0.8, 1.0001]
CAL_LABELS = ["< 50%", "50–80%", ">= 80%"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny markdown table writer (no extra dependency)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    int_cols = {c for c in df.columns if pd.api.types.is_integer_dtype(df[c])}
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            str(int(v)) if c in int_cols else (floatfmt.format(v) if isinstance(v, float) else str(v))
            for c, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def weights_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Check 1: last evening's weights plus min-max across all test evenings."""
    return pd.DataFrame({
        "last_evening": weights.iloc[-1],
        "min": weights.min(),
        "max": weights.max(),
    }).rename_axis("feature")


def calibration_table(pred: pd.DataFrame, col: str = "p_model") -> pd.DataFrame:
    """Check 2: group nights by predicted probability; compare with what happened."""
    groups = pd.cut(pred[col], CAL_BINS, labels=CAL_LABELS, right=False)
    t = pred.groupby(groups, observed=False).agg(
        nights=("truth", "size"), avg_predicted=(col, "mean"), actual_alert_rate=("truth", "mean"))
    t.index.name = "predicted"
    return t


def worst_nights(pred: pd.DataFrame, features: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Check 3: the k nights with the largest squared error of the model."""
    err = (pred["p_model"] - pred["truth"]) ** 2
    top = err.nlargest(k).index
    out = pred.loc[top, ["truth", "p_model", "p_baseline_B"]].copy()
    out["sq_error"] = err.loc[top]
    return out.join(features)


def halves(pred: pd.DataFrame) -> pd.DataFrame:
    """Check 4: test nights 1-60 vs 61-120; model vs B within each half."""
    rows = {}
    for name, part in [("nights 1–60", pred.iloc[:60]), ("nights 61–120", pred.iloc[60:])]:
        bm = ((part["p_model"] - part["truth"]) ** 2).mean()
        bb = ((part["p_baseline_B"] - part["truth"]) ** 2).mean()
        rows[f"{name} ({part.index[0].date()}..{part.index[-1].date()})"] = {
            "alert_rate": part["truth"].mean(), "avg_p_model": part["p_model"].mean(),
            "avg_p_B": part["p_baseline_B"].mean(), "brier_model": bm, "brier_B": bb, "skill_vs_B": 1 - bm / bb}
    return pd.DataFrame(rows).T.rename_axis("half")
