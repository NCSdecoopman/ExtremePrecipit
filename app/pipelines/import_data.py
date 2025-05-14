import streamlit as st

from app.utils.data_utils import load_data, cleaning_data_observed, dont_show_extreme, add_metadata, get_column_load
from app.utils.stats_utils import compute_statistic_per_point
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
    
    stat_choice_key, scale_choice_key, min_year_choice, max_year_choice, season_choice_key, missing_rate, quantile_choice = params
    loader = load_data_cached(use_cache)

    # Colonne de statistique nécessaire au chargement
    col_to_load, col_important = get_column_load(stat_choice_key, scale_choice_key)

    try:
        modelised_load = loader(
            'modelised', 'horaire',
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
            'observed', 'horaire' if scale_choice_key == 'mm_h' else 'quotidien',
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
