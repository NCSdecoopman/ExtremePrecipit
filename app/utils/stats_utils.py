import streamlit as st
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_statistic_per_point(df: pl.DataFrame, stat_key: str) -> pl.DataFrame:
    cols = df.columns

    if stat_key == "mean":
        has_h = "mean_mm_h" in cols
        if has_h:
            df = df.with_columns(
                (pl.col("mean_mm_h") * 24).alias("mean_mm_j")
            )
        return df.group_by("NUM_POSTE").agg([
            *( [pl.col("mean_mm_h").mean().alias("mean_all_mm_h")] if has_h else [] ),
            *( [pl.col("mean_mm_j").mean().alias("mean_all_mm_j")] if has_h else [] ),
        ])

    elif stat_key == "max":
        return df.group_by("NUM_POSTE").agg([
            *( [pl.col("max_mm_h").max().alias("max_all_mm_h")] if "max_mm_h" in cols else [] ),
            *( [pl.col("max_mm_j").max().alias("max_all_mm_j")] if "max_mm_j" in cols else [] ),
        ])

    elif stat_key == "mean-max":
        return df.group_by("NUM_POSTE").agg([
            *( [pl.col("max_mm_h").mean().alias("max_mean_mm_h")] if "max_mm_h" in cols else [] ),
            *( [pl.col("max_mm_j").mean().alias("max_mean_mm_j")] if "max_mm_j" in cols else [] ),
        ])

    elif stat_key == "date":
        res = []
        if "max_mm_h" in cols and "max_date_mm_h" in cols:
            df_h = (
                df.sort("max_mm_h", descending=True)
                .group_by("NUM_POSTE")
                .agg(pl.col("max_date_mm_h").first().alias("date_max_h"))
            )
            res.append(df_h)
        if "max_mm_j" in cols and "max_date_mm_j" in cols:
            df_j = (
                df.sort("max_mm_j", descending=True)
                .group_by("NUM_POSTE")
                .agg(pl.col("max_date_mm_j").first().alias("date_max_j"))
            )
            res.append(df_j)

        if not res:
            raise ValueError("Aucune date de maximum disponible.")
        elif len(res) == 1:
            return res[0]
        else:
            return res[0].join(res[1], on="NUM_POSTE", how="outer")

    elif stat_key == "month":
        exprs = []
        if "max_date_mm_h" in cols:
            exprs.append(
                pl.col("max_date_mm_h")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.month()
                .alias("mois_max_h")
            )
        if "max_date_mm_j" in cols:
            exprs.append(
                pl.col("max_date_mm_j")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.month()
                .alias("mois_max_j")
            )
        if not exprs:
            raise ValueError("Aucune date de maximum pour extraire les mois.")

        df = df.with_columns(exprs)

        mois_h = mois_j = None

        if "mois_max_h" in df.columns:
            mois_h = (
                df.drop_nulls("mois_max_h")
                .group_by(["NUM_POSTE", "mois_max_h"])
                .len()
                .sort(["NUM_POSTE", "len"], descending=[False, True])
                .unique(subset=["NUM_POSTE"])
                .select(["NUM_POSTE", "mois_max_h"])
                .rename({"mois_max_h": "mois_pluvieux_h"})
            )

        if "mois_max_j" in df.columns:
            mois_j = (
                df.drop_nulls("mois_max_j")
                .group_by(["NUM_POSTE", "mois_max_j"])
                .len()
                .sort(["NUM_POSTE", "len"], descending=[False, True])
                .unique(subset=["NUM_POSTE"])
                .select(["NUM_POSTE", "mois_max_j"])
                .rename({"mois_max_j": "mois_pluvieux_j"})
            )

        if mois_h is None and mois_j is None:
            return pl.DataFrame(schema={"NUM_POSTE": pl.Int64, "mois_pluvieux_h": pl.Int32, "mois_pluvieux_j": pl.Int32})
        elif mois_h is None:
            return mois_j.with_columns([pl.lit(None, dtype=pl.Int32).alias("mois_pluvieux_h")])
        elif mois_j is None:
            return mois_h.with_columns([pl.lit(None, dtype=pl.Int32).alias("mois_pluvieux_j")])
        else:
            return mois_h.join(mois_j, on="NUM_POSTE", how="outer")

    elif stat_key == "numday":
        if "n_days_gt1mm" not in df.columns:
            raise ValueError("Colonne `n_days_gt1mm` manquante.")
        return (
            df.group_by("NUM_POSTE")
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

    corr = np.corrcoef(x, y)[0, 1]  # coefficient de corrélation de Pearson
    r2_corr = corr**2  # son carré est toujours >= 0

    return me, mae, rmse, r2_corr


def compute_return_levels_ns(params: dict, T: np.ndarray, t_norm: float) -> np.ndarray:
    """
    Calcule les niveaux de retour selon le modèle NS-GEV fourni.
    - params : dictionnaire des paramètres GEV d'un point
    - T : périodes de retour (en années)
    - t_norm : covariable temporelle normalisée (ex : 0 pour année moyenne)
    """    
    mu = params.get("mu0", 0) + params["mu1"] * t_norm if "mu1" in params else params.get("mu0", 0) # μ(t)
    sigma = params.get("sigma0", 0) + params["sigma1"] * t_norm if "sigma1" in params else params.get("sigma0", 0) # σ(t)
    xi = params.get("xi", 0) # xi contant

    if xi != 0:
        qT = mu + (sigma / xi) * ((-np.log(1 - 1 / T))**(-xi) - 1)
    else:
        qT = mu - sigma * np.log(-np.log(1 - 1/T))

    return qT