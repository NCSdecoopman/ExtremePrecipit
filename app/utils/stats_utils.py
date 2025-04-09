import polars as pl

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
            pl.col("max_date_mm_h").str.strptime(pl.Date, strict=False).dt.month().alias("mois_max_h"),
            pl.col("max_date_mm_j").str.strptime(pl.Date, strict=False).dt.month().alias("mois_max_j")
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

        return mois_h.join(mois_j, on=["lat", "lon"], how="outer")

    elif stat_key == "numday":
        return (
            df.group_by(["lat", "lon"])
            .agg(pl.col("n_days_gt1mm").mean().alias("jours_pluie_moyen"))
        )

    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")
