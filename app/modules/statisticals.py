import pandas as pd
import streamlit as st
import calendar

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *

import pydeck as pdk

st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)

def compute_statistic_per_point(df: pd.DataFrame, stat_key: str, min_year: int = None, max_year: int = None, months: tuple = None) -> pd.DataFrame:
    with st.spinner("Chargement des fichiers..."):
        progress_bar = st.progress(0)
    
    if stat_key == "mean":
        # Calcul du nombre de jours sur la période sélectionnée
        total_days = 0
        for year in range(min_year, max_year + 1):
            for month in months:
                try:
                    days_in_month = calendar.monthrange(year, month)[1]
                    total_days += days_in_month
                except:
                    continue  # en cas de mauvais mois/année

        if total_days == 0:
            raise ValueError("Aucun jour trouvé dans la période sélectionnée.")

        # Somme des précipitations cumulées
        result = df.groupby(["lat", "lon"])["sum_mm"].sum().reset_index(name="sum_mm_total")

        # Moyenne journalière sur la période
        result["mean_mm_j"] = result["sum_mm_total"] / total_days
        result["mean_mm_h"] = result["mean_mm_j"] / 24

        return result[["lat", "lon", "mean_mm_h", "mean_mm_j"]]

    elif stat_key == "max":
        return df.groupby(["lat", "lon"]).agg(
            max_all_mm_h=("max_mm_h", "max"),
            max_all_mm_j=("max_mm_j", "max")
        ).reset_index()

    elif stat_key == "mean-max":
        return df.groupby(["lat", "lon"]).agg(
            max_mean_mm_h=("max_mm_h", "mean"),
            max_mean_mm_j=("max_mm_j", "mean")
        ).reset_index()

    elif stat_key == "date":
        # Date du maximum horaire et journalier
        idx_h = df.groupby(["lat", "lon"])["max_mm_h"].idxmax()
        idx_j = df.groupby(["lat", "lon"])["max_mm_j"].idxmax()

        df_h = df.loc[idx_h, ["lat", "lon", "max_date_mm_h"]].rename(columns={"max_date_mm_h": "date_max_h"})
        df_j = df.loc[idx_j, ["lat", "lon", "max_date_mm_j"]].rename(columns={"max_date_mm_j": "date_max_j"})

        return pd.merge(df_h, df_j, on=["lat", "lon"])

    elif stat_key == "month":
        # Mois le plus fréquent d'occurrence des maximas horaires et journaliers
        df["mois_max_h"] = df["max_date_mm_h"].str[5:7].astype(int)
        df["mois_max_j"] = df["max_date_mm_j"].str[5:7].astype(int)

        mois_h = df.groupby(["lat", "lon"])["mois_max_h"] \
                .agg(lambda x: x.value_counts().idxmax()) \
                .reset_index(name="mois_pluvieux_h")

        mois_j = df.groupby(["lat", "lon"])["mois_max_j"] \
                .agg(lambda x: x.value_counts().idxmax()) \
                .reset_index(name="mois_pluvieux_j")

        return pd.merge(mois_h, mois_j, on=["lat", "lon"])


    elif stat_key == "numday":
        # Moyenne du nombre de jours de pluie sur la période sélectionnée
        n_years = df["max_date_mm_h"].str[:4].astype(int).nunique()
        return df.groupby(["lat", "lon"])["n_days_gt1mm"] \
                 .sum() \
                 .div(n_years) \
                 .reset_index(name="jours_pluie_moyen")

    else:
        raise ValueError(f"Statistique inconnue : {stat_key}")
    
    progress_bar.empty()  # Efface la barre de progression
    

def show(config_path):

    config = load_config(config_path)

    STATS, SEASON, SCALE = menu_config()

    min_years = config["years"]["min"]
    max_years = config["years"]["max"]

    stat_choice, min_year_choice, max_year_choice, season_choice, scale_choice = menu_statisticals(
        min_years,
        max_years,
        STATS,
        SEASON
    )

    stat_choice_key = STATS[stat_choice]
    season_choice_key = SEASON[season_choice]
    scale_choice_key = SCALE[scale_choice]

    try:
        df_all = load_arome_data(
            min_year_choice,
            max_year_choice,
            tuple(season_choice_key),  # tuple hashable
            config  # utilisé pour le hash automatique de cache
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return

    # Calcul des statistiques
    result_df = compute_statistic_per_point(df_all, stat_choice_key, min_year_choice, max_year_choice, season_choice_key)
    column_to_show = get_stat_column_name(stat_choice_key, scale_choice_key)
   
    # Définir l'échelle personnalisée continue
    colormap = echelle_config("continu")

    # Normalisation de la légende
    result_df, vmin, vmax = formalised_legend(result_df, column_to_show, colormap)

    # Créer le layer Pydeck
    layer = create_layer(result_df)

    # Tooltip
    tooltip = create_tooltip(stat_choice)

    # View de la carte
    view_state = pdk.ViewState(latitude=46.6, longitude=2.2, zoom=4.5)

    col1, col2, col3 = st.columns([2.5, 1, 3.5])  # Carte large, légende étroite
    height = 600

    with col1:
        plot_map(layer, view_state, tooltip)

    with col2:
        unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
        display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=8, label=unit_label)       
