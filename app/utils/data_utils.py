import numpy as np
import polars as pl
import streamlit as st
from scipy.spatial import cKDTree

from app.utils.config_utils import menu_config_statisticals

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
    _, SEASON, _ = menu_config_statisticals()
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

            # Ajout de la colonne year
            df = df.with_columns(pl.lit(year).alias("year"))
            
            dataframes.append(df)

        except Exception as e:
            errors.append(f"{year} ({season}) : {e}")

    if errors:
        for err in errors:
            st.warning(f"Erreur : {err}")

    if not dataframes:
        raise ValueError("Aucune donnée chargée.")
    
    return pl.concat(dataframes, how="vertical")


def cleaning_data_observed(
    df: pl.DataFrame,
    len_serie: float = None,
    nan_limit: float = 0.10
) -> pl.DataFrame:
    """
    Filtre les maxima par deux critères :
      1) on annule les valeurs d’une année si nan_ratio > nan_limit
      2) on ne garde que les stations ayant au moins n années valides
    """
    # ——— règles dépendant de l’échelle ———
    if len_serie is None:
        raise ValueError('Paramètre len_serie à préciser')
    
    # Selection des saisons avec nan_limit au maximum
    df_filter = df.filter(pl.col("nan_ratio") <= nan_limit)

    # Calcul du nombre d'années valides par station NUM_POSTE
    station_counts = (
        df_filter.group_by("NUM_POSTE")
        .agg(pl.col("year").n_unique().alias("num_years"))
    )

    # Sélection des NUM_POSTE avec au moins len_serie d'années valides
    valid_stations = station_counts.filter(pl.col("num_years") >= len_serie)

    # Jointure pour ne garder que les stations valides
    df_final = df_filter.filter(
        pl.col("NUM_POSTE").is_in(valid_stations["NUM_POSTE"])
    )

    return df_final

def dont_show_extreme(
    modelised: pl.DataFrame,
    observed:   pl.DataFrame,
    column:     str,
    quantile_choice: float,
    stat_choice_key: str = None
) -> tuple[pl.DataFrame, pl.DataFrame]:

    if stat_choice_key not in ("month", "date"):
        # 1) Calcul des quantiles
        q_mod = modelised.select(
            pl.col(column).quantile(quantile_choice, interpolation="nearest")
        ).item()

        if observed is None or observed.height == 0:
            seuil = q_mod
        else:
            q_obs = observed.select(
                pl.col(column).quantile(quantile_choice, interpolation="nearest")
            ).item()
            seuil = max(q_mod, q_obs)

        # 2) Saturation des couleurs
        clamp_expr = (
            pl.when(pl.col(column).abs() > seuil)
              .then(pl.lit(seuil) * pl.col(column).sign())
              .otherwise(pl.col(column))
              .alias(column)
        )

        # 3) Renvoi des tableaux
        modelised_show = modelised.with_columns(clamp_expr)
        observed_show  = observed.with_columns(clamp_expr)

    else:
        modelised_show, observed_show = modelised, observed

    return modelised_show, observed_show


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
        .join(obs.select(["NUM_POSTE_obs", "lat", "lon", column_to_show]), on="NUM_POSTE_obs", how="left")
        .join(mod.select(["NUM_POSTE_mod", column_to_show]), on="NUM_POSTE_mod", how="left", suffix="_mod")
        .rename({column_to_show: "Station", f"{column_to_show}_mod": "AROME"})
    )
    matched = matched.select(["NUM_POSTE_obs", "lat", "lon", "NUM_POSTE_mod", "Station", "AROME"]).drop_nulls()
    
    return matched


def standardize_year(year: float, min_year: int, max_year: int) -> float:
    """
    Normalise une année `year` entre 0 et 1 avec une transformation min-max.
    """
    return (year - min_year) / (max_year - min_year)


def filter_nan(df: pl.DataFrame, columns: list[str]):
    return df.drop_nulls(subset=columns)