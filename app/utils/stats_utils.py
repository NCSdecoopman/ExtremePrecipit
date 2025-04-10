import streamlit as st
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_statistic_per_point(df: pl.DataFrame, stat_key: str) -> pl.DataFrame:
    if stat_key == "mean":
        df = df.with_columns(
            (pl.col("mean_mm_h") * 24).alias("mean_mm_j")
        )
        return (
            df.group_by(["lat", "lon"])
            .agg([
                pl.col("mean_mm_h").mean().alias("mean_all_mm_h"),
                pl.col("mean_mm_j").mean().alias("mean_all_mm_j"),
            ])
        )

    elif stat_key == "max":
        return (
            df.group_by(["lat", "lon"])
            .agg([
                pl.col("max_mm_h").max().alias("max_all_mm_h"),
                pl.col("max_mm_j").max().alias("max_all_mm_j"),
            ])
        )

    elif stat_key == "mean-max":
        return (
            df.group_by(["lat", "lon"])
            .agg([
                pl.col("max_mm_h").mean().alias("max_mean_mm_h"),
                pl.col("max_mm_j").mean().alias("max_mean_mm_j"),
            ])
        )

    elif stat_key == "date":
        # Max horaire
        df_h = (
            df.sort("max_mm_h", descending=True)
            .group_by(["lat", "lon"])
            .agg(pl.col("max_date_mm_h").first().alias("date_max_h"))
        )

        # Max journalier
        df_j = (
            df.sort("max_mm_j", descending=True)
            .group_by(["lat", "lon"])
            .agg(pl.col("max_date_mm_j").first().alias("date_max_j"))
        )

        return df_h.join(df_j, on=["lat", "lon"], how="outer")

    elif stat_key == "month":
        df = df.with_columns([
            pl.col("max_date_mm_h")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
            .dt.month()
            .alias("mois_max_h"),

            pl.col("max_date_mm_j")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
            .dt.month()
            .alias("mois_max_j"),
        ])
        # Horaire : mois le plus fréquent
        mois_h = (
            df.drop_nulls(["mois_max_h"])
            .group_by(["lat", "lon", "mois_max_h"])
            .len()
            .sort(["lat", "lon", "len"], descending=[False, False, True])
            .unique(subset=["lat", "lon"])
            .select(["lat", "lon", "mois_max_h"])
            .rename({"mois_max_h": "mois_pluvieux_h"})
        )

        # Journalier : idem
        mois_j = (
            df.drop_nulls(["mois_max_j"])
            .group_by(["lat", "lon", "mois_max_j"])
            .len()
            .sort(["lat", "lon", "len"], descending=[False, False, True])
            .unique(subset=["lat", "lon"])
            .select(["lat", "lon", "mois_max_j"])
            .rename({"mois_max_j": "mois_pluvieux_j"})
        )

        mois_h = mois_h.with_columns([
            pl.col("lat").cast(pl.Float32),
            pl.col("lon").cast(pl.Float32)
        ])
        mois_j = mois_j.with_columns([
            pl.col("lat").cast(pl.Float32),
            pl.col("lon").cast(pl.Float32)
        ])

        if mois_h.is_empty() and mois_j.is_empty():
            return pl.DataFrame(schema={"lat": pl.Float32, "lon": pl.Float32, "mois_pluvieux_h": pl.Int32, "mois_pluvieux_j": pl.Int32})
        elif mois_h.is_empty():
            return mois_j.with_columns([
                pl.lit(None, dtype=pl.Int32).alias("mois_pluvieux_h")
            ])
        elif mois_j.is_empty():
            return mois_h.with_columns([
                pl.lit(None, dtype=pl.Int32).alias("mois_pluvieux_j")
            ])
        else:
            return mois_h.join(mois_j, on=["lat", "lon"], how="outer")

    elif stat_key == "numday":
        return (
            df.group_by(["lat", "lon"])
            .agg(pl.col("n_days_gt1mm").mean().alias("jours_pluie_moyen"))
        )

    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")


def generate_metrics(df: pl.DataFrame, x_label: str = "pr_mod", y_label: str = "pr_obs"):
    x = df[x_label].to_numpy()
    y = df[y_label].to_numpy()

    if len(x) != len(y):
        st.error("Longueur x et y différente")

    rmse = np.sqrt(mean_squared_error(y, x))
    mae = mean_absolute_error(y, x)
    me = np.mean(x - y)
    r2 = r2_score(y, x)

    return me, mae, rmse, r2