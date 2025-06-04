import streamlit as st

from app.utils.data_utils import (
    load_data, 
    cleaning_data_observed, 
    dont_show_extreme, 
    add_metadata, 
    get_column_load,
    filter_nan
)
from app.utils.stats_utils import compute_statistic_per_point
from app.utils.gev_utils import safe_compute_return_df, compute_delta_qT, compute_delta_stat
from app.utils.legends_utils import get_stat_column_name

import polars as pl

def load_data_cached(use_cache: bool):
    if use_cache:
        return st.cache_data(load_data_inner) # Version cachée qui retourne un DataFrame pour la sérialisation.
    else:
        return load_data_inner

def load_data_inner(type_data: str, echelle: str, min_year: int, max_year: int, season_key: str, col_to_load: list, config) -> pl.DataFrame:
    return load_data(type_data, echelle, min_year, max_year, season_key, col_to_load, config)


def pipeline_data(params, config, use_cache=False):
    
    stat_choice_key, scale_choice_key, min_year_choice, max_year_choice, season_choice_key, missing_rate, quantile_choice, scale_choice = params
    loader = load_data_cached(use_cache)

    # Colonne de statistique nécessaire au chargement
    col_to_load, col_important = get_column_load(stat_choice_key, scale_choice_key)

    if scale_choice == "Journalière":
        scale_choice = "quotidien"
    elif scale_choice == "Horaire":
        scale_choice = "horaire"

    try:
        modelised_load = loader(
            'modelised', scale_choice if scale_choice != "quotidien" else "horaire",
            min_year_choice,
            max_year_choice,
            season_choice_key,
            col_to_load,
            config
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des données modélisées : {e}")

    try:
        observed_load = loader(
            'observed', scale_choice,
            min_year_choice,
            max_year_choice,
            season_choice_key,
            col_to_load + ["nan_ratio"],
            config
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des données observées : {e}")

    # Selection des données observées
    df_observed_cleaning = cleaning_data_observed(observed_load, missing_rate)
    
    # Calcul des statistiques
    modelised = compute_statistic_per_point(modelised_load, stat_choice_key)
    observed = compute_statistic_per_point(df_observed_cleaning, stat_choice_key)
   
    # Ajout de l'altitude et des lat lon
    modelised = add_metadata(modelised, scale_choice_key, type='modelised')    
    observed = add_metadata(observed, scale_choice_key, type='observed')    

    # Obtention de la colonne étudiée
    column = get_stat_column_name(stat_choice_key, scale_choice_key)

    # Retrait des extrêmes pour l'affichage uniquement
    modelised_show = dont_show_extreme(modelised, column, quantile_choice, stat_choice_key)
    observed_show = dont_show_extreme(observed, column, quantile_choice, stat_choice_key)

    return {
        "modelised_load": modelised_load,
        "observed_load": observed_load,
        "observed_cleaning": df_observed_cleaning,
        "modelised_show": modelised_show,
        "observed_show": observed_show,
        "modelised": modelised,
        "observed": observed,
        "column": column
    }

def pipeline_data_gev(params):

    column = params["param_choice"]

    BOOTSTRAP = False
    if "_bootstrap" in params['model_name']: # dans le cas des modèles avec bootstrap
        BOOTSTRAP = True
        # On repasse sur les fichiers non boostrapés
        params['model_name'] = params['model_name'].replace('_bootstrap', '')

    df_modelised_load = pl.read_parquet(params["mod_dir"] / f"gev_param_{params['model_name']}.parquet")
    df_observed_load = pl.read_parquet(params["obs_dir"] / f"gev_param_{params['model_name']}.parquet")

    df_modelised = filter_nan(df_modelised_load, "xi") # xi est toujours valable   
    df_observed = filter_nan(df_observed_load, "xi") # xi est toujours valable

    df_modelised = add_metadata(df_modelised, "mm_h" if params["echelle"] == "horaire" else "mm_j", type="modelised")
    df_observed = add_metadata(df_observed, "mm_h" if params["echelle"] == "horaire" else "mm_j", type="observed")       
    
    # Étape 1 : créer une colonne avec les paramètres nettoyés
    df_modelised = safe_compute_return_df(df_modelised)
    df_observed = safe_compute_return_df(df_observed)

    # Étape 2 : appliquer delta_qT_decennale (avec numpy)
    T_choice = params["T_choice"]  # ou récupéré dynamiquement via Streamlit

    if "_break_year" in params['model_name']: # dans le cas des modèles avec point de rupture
        year_range = params["max_year_choice"] - params["config"]["years"]["rupture"] # Δa+ = a_max - a_rupture
    else:
        year_range = params["max_year_choice"] - params["min_year_choice"] # Δa = a_max - a_min


    if column == "Δqᵀ":
        # Calcul du delta qT
        df_modelised = df_modelised.with_columns([
            pl.struct(["mu1", "sigma1", "xi"])
            .map_elements(lambda row: compute_delta_qT(row, T_choice, year_range, params["par_X_annees"]), return_dtype=pl.Float64)
            .alias("Δqᵀ")
        ])

        df_observed = df_observed.with_columns([
            pl.struct(["mu1", "sigma1", "xi"])
            .map_elements(lambda row: compute_delta_qT(row, T_choice, year_range, params["par_X_annees"]), return_dtype=pl.Float64)
            .alias("Δqᵀ")
        ])


    elif column in ["ΔE", "ΔVar", "ΔCV"]:
        t_start = params["min_year_choice"]
        t_end = params["max_year_choice"]
        t0 = params["config"]["years"]["rupture"]

        df_modelised = df_modelised.with_columns([
            pl.struct(["mu0", "mu1", "sigma0", "sigma1", "xi"])
            .map_elements(lambda row: compute_delta_stat(row, column, t_start, t0 , t_end, params["par_X_annees"]), return_dtype=pl.Float64)
            .alias(column)
        ])

        df_observed = df_observed.with_columns([
            pl.struct(["mu0", "mu1", "sigma0", "sigma1", "xi"])
            .map_elements(lambda row: compute_delta_stat(row, column, t_start, t0, t_end, params["par_X_annees"]), return_dtype=pl.Float64)
            .alias(column)
        ])



    if BOOTSTRAP:
        df_mod_bootstrap = pl.read_parquet(params["mod_dir"] / f"gev_param_{params['model_name']}_bootstrap.parquet")
        df_obs_bootstrap = pl.read_parquet(params["obs_dir"] / f"gev_param_{params['model_name']}_bootstrap.parquet")

        # Recalcule delta_qT pour chaque bootstrap
        df_mod_bootstrap = df_mod_bootstrap.with_columns([
            pl.struct(["mu1", "sigma1", "xi"]).map_elements(
                lambda row: compute_delta_qT(
                    row,
                    params["T_choice"],
                    year_range,
                    params["par_X_annees"]
                ),
                return_dtype=pl.Float64
            ).alias("Δqᵀ")
        ])

        df_obs_bootstrap = df_obs_bootstrap.with_columns([
            pl.struct(["mu1", "sigma1", "xi"]).map_elements(
                lambda row: compute_delta_qT(
                    row,
                    params["T_choice"],
                    year_range,
                    params["par_X_annees"]
                ),
                return_dtype=pl.Float64
            ).alias("Δqᵀ")
        ])

        # Calcule les bornes de l'intervalle de confiance
        df_ic_mod = (
            df_mod_bootstrap
            .group_by("NUM_POSTE")
            .agg([
                pl.col("Δqᵀ").quantile(0.05, "nearest").alias("Δqᵀ_q050"),
                pl.col("Δqᵀ").quantile(0.95, "nearest").alias("Δqᵀ_q950"),
            ])
        )

        df_ic_obs = (
            df_obs_bootstrap
            .group_by("NUM_POSTE")
            .agg([
                pl.col("Δqᵀ").quantile(0.05, "nearest").alias("Δqᵀ_q050"),
                pl.col("Δqᵀ").quantile(0.95, "nearest").alias("Δqᵀ_q950"),
            ])
        )

        # Forcer NUM_POSTE à être de même type (int) dans les deux DataFrames
        df_ic_mod = df_ic_mod.with_columns([pl.col("NUM_POSTE").cast(pl.Int64)])
        df_ic_obs = df_ic_obs.with_columns([pl.col("NUM_POSTE").cast(pl.Int64)])
        df_modelised = df_modelised.with_columns([pl.col("NUM_POSTE").cast(pl.Int64)])
        df_observed = df_observed.with_columns([pl.col("NUM_POSTE").cast(pl.Int64)])


        # Join à df_observed
        df_modelised = df_modelised.join(df_ic_mod, on="NUM_POSTE", how="left")
        df_observed = df_observed.join(df_ic_obs, on="NUM_POSTE", how="left")

        # Création d'une colonne est significatif ou non (ne recoupe pas l'intervalle)
        df_modelised = df_modelised.with_columns([
            (
                ~((pl.col("Δqᵀ_q050") <= 0) & (pl.col("Δqᵀ_q950") >= 0))
            ).alias("is_significant")
        ])

        df_observed = df_observed.with_columns([
            (
                ~((pl.col("Δqᵀ_q050") <= 0) & (pl.col("Δqᵀ_q950") >= 0))
            ).alias("is_significant")
        ])

    # Retrait des percentiles
    modelised_show = dont_show_extreme(df_modelised, column, params["quantile_choice"])
    observed_show = dont_show_extreme(df_observed, column, params["quantile_choice"])
   
    if column in ["Δqᵀ", "ΔE", "ΔVar", "ΔCV"]:

        val_max = max(modelised_show[column].max(), observed_show[column].max())
        val_min = min(modelised_show[column].min(), observed_show[column].min())
        abs_max = max(abs(val_min), abs(val_max))

        return {
            "modelised_load": df_modelised_load,
            "observed_load": df_observed_load,
            "modelised": df_modelised,
            "observed": df_observed,
            "modelised_show": modelised_show,
            "observed_show": observed_show, 
            "column": column,
            "vmin": -abs_max,
            "vmax": abs_max,
            "echelle": "diverging_zero_white",
            "continu": True
        }
    
    else:

        return {
            "modelised_load": df_modelised_load,
            "observed_load": df_observed_load,
            "modelised": df_modelised,
            "observed": df_observed,
            "modelised_show": modelised_show,
            "observed_show": observed_show, 
            "column": column
        }