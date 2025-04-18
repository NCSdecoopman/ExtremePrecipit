import polars as pl
import streamlit as st
from scipy.spatial import cKDTree

from app.utils.config_utils import menu_config

def get_column_load(stat: str, scale: str):
    if stat == "mean":
        col = "mean_mm_h"
    elif stat == "max":
        col = f"max_{scale}"
    elif stat == "mean-max":
        col = f"max_{scale}"
    elif stat == "month":
        col = f"max_date_{scale}"
    elif stat == "numday":
        col = "n_days_gt1mm"
    else:
        raise ValueError(f"Stat '{stat}' is not recognized")

    return ["NUM_POSTE", col], col

def load_season(year: int, season_key: str, base_path: str, col_to_load: str) -> pl.DataFrame:
    filename = f"{base_path}/{year:04d}/{season_key}.parquet"
    return pl.read_parquet(filename, columns=col_to_load)

def load_data(type_data: str, echelle: str, min_year: int, max_year: int, season: str, col_to_load: str, config) -> pl.DataFrame:
    _, SEASON, _ = menu_config()
    if season not in SEASON.values():
        raise ValueError(f"Saison inconnue : {season}")

    base_path = f'{config["statisticals"][type_data]}/{echelle}'

    dataframes = []
    errors = []

    for year in range(min_year, max_year + 1):
        try:
            df = load_season(year, season, base_path, col_to_load)
            # Conversion explicite des colonnes dates uniquement si elles existent
            for col in ["max_date_mm_h", "max_date_mm_j"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)  # s'assure qu'on peut parser avec str.strptime
                        .str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
                        .cast(pl.Utf8)  # retour sous forme de string (comme dans l'ancien code Pandas)
                    )

            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")

    return pl.concat(dataframes, how="vertical")


def cleaning_data_observed(df: pl.DataFrame, nan_limit: float = 0.1) -> pl.DataFrame:
    # Moyenne du ratio de NaN par station (lat, lon)
    station_counts = (
        df.group_by(["NUM_POSTE"])
        .agg(pl.col("nan_ratio").mean().alias("nan_ratio"))
    )

    # Stations valides selon le seuil
    valid = station_counts.filter(pl.col("nan_ratio") <= nan_limit)

    # Jointure pour ne garder que les stations valides
    df_filtered = df.join(valid.select(["NUM_POSTE"]), on=["NUM_POSTE"], how="inner")

    return df_filtered

def add_metadata(df: pl.DataFrame, scale: str, type: str) -> pl.DataFrame:
    echelle = 'horaire' if scale == 'mm_h' else 'quotidien'
    
    # Charger les metadonnées avec Polars
    df_meta = pl.read_csv(f"data/metadonnees/{type}/postes_{echelle}.csv")
    # Harmoniser les types des colonnes lat/lon des deux côtés
    df_meta = df_meta.with_columns([
        pl.col("NUM_POSTE").cast(pl.Int32),
        pl.col("lat").cast(pl.Float32),
        pl.col("lon").cast(pl.Float32),
        pl.col("altitude").cast(pl.Int32)  # altitude en entier
    ])

    df = df.with_columns([  # forcer ici aussi
        pl.col("NUM_POSTE").cast(pl.Int32)
    ])

    # Join sur NUM_POSTE
    return df.join(df_meta, on=["NUM_POSTE"], how="left")


def find_matching_point(df_model: pl.DataFrame, lat_obs: float, lon_obs: float):
    df_model = df_model.with_columns([
        ((pl.col("lat") - lat_obs) ** 2 + (pl.col("lon") - lon_obs) ** 2).sqrt().alias("dist")
    ])
    closest_row = df_model.filter(pl.col("dist") == pl.col("dist").min()).select(["lat", "lon"]).row(0)
    return closest_row  # (lat, lon)

def match_and_compare(
    obs_df: pl.DataFrame,
    mod_df: pl.DataFrame,
    column_to_show: str,
    obs_vs_mod: pl.DataFrame = None
) -> pl.DataFrame:
    
    if obs_vs_mod is None:
        raise ValueError("obs_vs_mod must be provided with NUM_POSTE_obs and NUM_POSTE_mod columns")
    
    obs_vs_mod = obs_vs_mod.with_columns(
        pl.col("NUM_POSTE_obs").cast(pl.Int32)
    ).filter(
        pl.col("NUM_POSTE_obs").is_in(obs_df["NUM_POSTE"].cast(pl.Int32))
    )

    # Renommer temporairement pour le join
    obs = obs_df.rename({"NUM_POSTE": "NUM_POSTE_obs"})
    mod = mod_df.rename({"NUM_POSTE": "NUM_POSTE_mod"})

    obs = obs_df.with_columns(
        pl.col("NUM_POSTE").cast(pl.Int32)
    ).rename({"NUM_POSTE": "NUM_POSTE_obs"})

    mod = mod_df.with_columns(
        pl.col("NUM_POSTE").cast(pl.Int32)
    ).rename({"NUM_POSTE": "NUM_POSTE_mod"})

    obs_vs_mod = obs_vs_mod.with_columns(
        pl.col("NUM_POSTE_obs").cast(pl.Int32),
        pl.col("NUM_POSTE_mod").cast(pl.Int32)
    )


    # Ajoute les valeurs observées et simulées en fonction des correspondances
    matched = (
        obs_vs_mod
        .join(obs.select(["NUM_POSTE_obs", column_to_show]), on="NUM_POSTE_obs", how="left")
        .join(mod.select(["NUM_POSTE_mod", column_to_show]), on="NUM_POSTE_mod", how="left", suffix="_mod")
        .rename({column_to_show: "pr_obs", f"{column_to_show}_mod": "pr_mod"})
    )
    matched = matched.select(["pr_obs", "pr_mod"]).drop_nulls()
    st.write(f"Points comparés : {matched.shape[0]}")
    return matched
