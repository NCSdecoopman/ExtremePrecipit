from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metrics_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "season" in df.columns:
        df = df.copy()
        df["season"] = df["season"].str.lower()
    return df


def season_row(df: pd.DataFrame, season: str) -> pd.Series:
    rows = df.loc[df["season"] == season.lower()]
    if rows.empty:
        raise ValueError(f"No metrics row for season={season!r}")
    if len(rows) > 1:
        raise ValueError(f"Multiple metrics rows for season={season!r}")
    return rows.iloc[0]


def read_map_metrics(path: str | Path, season: str) -> tuple[float, int, float, float]:
    row = season_row(load_metrics_csv(path), season)
    return float(row["r"]), int(row["n"]), float(row["me"]), float(row["delta"])
