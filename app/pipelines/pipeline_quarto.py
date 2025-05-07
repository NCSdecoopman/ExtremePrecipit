from pathlib import Path

import numpy as np
import polars as pl

from app.utils.map_utils import create_layer, create_scatter_layer, create_tooltip, plot_map
from app.utils.config_utils import echelle_config
from app.utils.legends_utils import formalised_legend, display_vertical_color_legend

import pydeck as pdk
import plotly.io as pio

from app.utils.data_utils import (
    load_data,
    cleaning_data_observed,
    get_column_load,
    match_and_compare,
    add_metadata,
    dont_show_extreme
)
from app.utils.stats_utils import (
    compute_statistic_per_point,
    generate_metrics
)
from app.utils.legends_utils import get_stat_column_name
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive


def data_utils(
    stat_choice_key: str,
    scale_choice_key: str,
    min_year_choice: int,
    max_year_choice: int,
    season_choice_key: str,
    missing_rate: float,
    quantile_choice: float,
    config: dict
):
    col_to_load, _ = get_column_load(stat_choice_key, scale_choice_key)

    modelised = load_data(
        'modelised', 'horaire',
        min_year_choice, max_year_choice,
        season_choice_key, col_to_load, config
    )

    observed = load_data(
        'observed',
        'horaire' if scale_choice_key == 'mm_h' else 'quotidien',
        min_year_choice, max_year_choice,
        season_choice_key, col_to_load + ["nan_ratio"], config
    )

    observed_cleaned = cleaning_data_observed(observed, missing_rate)

    modelised_stat = compute_statistic_per_point(modelised, stat_choice_key)
    observed_stat = compute_statistic_per_point(observed_cleaned, stat_choice_key)
    
    modelised_stat = add_metadata(modelised_stat, scale_choice_key, type='modelised')
    observed_stat = add_metadata(observed_stat, scale_choice_key, type='observed')
    
    column = get_stat_column_name(stat_choice_key, scale_choice_key)
    modelised_show = dont_show_extreme(modelised_stat, column, quantile_choice, stat_choice_key)
    observed_show = dont_show_extreme(observed_stat, column, quantile_choice, stat_choice_key)

    return {
        "modelised": modelised_stat,
        "observed": observed_stat,
        "modelised_show": modelised_show,
        "observed_show": observed_show,
        "column": column
    }

def pipeline_data_quarto(
        config: dict,
        stat_choice: str,
        echelle: str,
        min_year: int,
        max_year: int,
        season_choice: str,
        unit_choice: str = None,
        missing_choice: int = 0.15,
        quantile_choice: int = 0.999
    ):
    
    # Définir l'unité par défaut si non spécifiée
    if unit_choice is None:
        unit_choice = "mm/j" if echelle == "quotidien" else "mm/h"

    # Déterminer la clé de l’échelle à partir de l’unité
    scale_map = {
        "mm/j": "mm_j",
        "mm/h": "mm_h"
    }
    scale_choice_key = scale_map.get(unit_choice, "")

    # Adapter l'année minimale si l'échelle est saisonnière
    if echelle in {"hydro", "djf"}:
        min_year += 1

    return data_utils(
        stat_choice_key=stat_choice, # Ex : "mean"
        scale_choice_key=scale_choice_key,
        min_year_choice=min_year,
        max_year_choice=max_year,
        season_choice_key=season_choice,
        missing_rate=missing_choice,
        quantile_choice=quantile_choice,
        config=config
    )

def pipeline_map_quarto(
    column: str,
    result: dict,
    unit_label: str,
    height: int = 500,
    continu: bool = True,
    categories=None,
    n_colors: int = 15
):
    """
    Génère une carte Pydeck et une légende HTML avec une normalisation commune pour
    les ensembles de données 'modelised_show' et 'observed'.
    """
    # Normalisation des couleurs sur les données modélisées
    colormap = echelle_config("continu" if continu else "discret", n_colors=n_colors)

    # Normalisation des valeurs modélisées
    result_df_modelised_show, vmin_mod, vmax_mod = formalised_legend(
        result["modelised_show"], 
        column_to_show=column, 
        colormap=colormap,
        is_categorical=not continu,
        categories=categories
    )

    # Normalisation des observations avec les mêmes bornes
    result_df_observed_show, vmin_obs, vmax_obs = formalised_legend(
        result["observed_show"], 
        column_to_show=column, 
        colormap=colormap,
        is_categorical=not continu,
        categories=categories
    )

    # Calcul des bornes communes
    vmin_commun = min(vmin_mod, vmin_obs)
    vmax_commun = max(vmax_mod, vmax_obs)

    # Mise à jour de la normalisation pour les deux ensembles de données avec les bornes communes
    result_df_modelised_show, _, _ = formalised_legend(
        result["modelised_show"], 
        column_to_show=column, 
        colormap=colormap,
        vmin=vmin_commun,
        vmax=vmax_commun,
        is_categorical=not continu,
        categories=categories
    )

    result_df_observed_show, _, _ = formalised_legend(
        result["observed_show"], 
        column_to_show=column, 
        colormap=colormap,
        vmin=vmin_commun,
        vmax=vmax_commun,
        is_categorical=not continu,
        categories=categories
    )

    # Création du layer modélisé et observé
    layer = create_layer(result_df_modelised_show)
    scatter_layer = create_scatter_layer(result_df_observed_show)

    # Tooltip et vue
    tooltip = create_tooltip(unit_label)
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=4.65)

    # Carte Pydeck
    deck = plot_map([layer, scatter_layer], view_state, tooltip)

    # Légende commune
    legend = display_vertical_color_legend(
        height, 
        colormap, 
        vmin_commun, 
        vmax_commun, 
        n_ticks=n_colors, 
        label=unit_label,
        model_labels=categories
    )


    return deck, legend



def pipeline_map_legend_scatter(
    name: str,
    result: dict,
    echelle: str,
    stat_choice_label: str,
    unit_choice: str = None,
    missing_choice: int = 0.15,
    height: int = 500):

    if unit_choice == None:
        unit_choice = "mm/j" if echelle == "quotidien" else "mm/h"

    Path("assets").mkdir(exist_ok=True)

    deck_path = f"assets/deck_map_{name}.html"
    scatter_path = f"assets/scatter_plot_{name}.html"

    deck, legend = pipeline_map_quarto(
        column=result["column"],
        result=result,
        unit_label=unit_choice,
        height=height
    )
    # Enregistrement de la carte
    deck.to_html(deck_path, notebook_display=False)

    # Affichage côte à côte
    html_map_legend = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; margin-top: 10px;">
        <iframe loading="lazy" src="{deck_path}" height="{height}" frameborder="0" style="flex: 3; width: 0; min-width: 0; max-width: 100%;"></iframe>
        <div style="flex: 1; max-width: 220px; margin-left: 5px;">{legend}</div>
    </div>
    """

    # Scatter plot
    df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
    obs_vs_mod = match_and_compare(result["observed"], result["modelised"], result["column"], df_obs_vs_mod)

    me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
 
    fig_scatter = generate_scatter_plot_interactive(
        df=obs_vs_mod, 
        stat_choice=stat_choice_label,
        unit_label=unit_choice, 
        height=height-60
    )
 
    fig_scatter.update_layout(
        template="simple_white",
        margin=dict(l=100, r=0, t=50, b=50),
        xaxis=dict(title=dict(text=f"AROME ({unit_choice})"), showticklabels=True),
        yaxis=dict(title=dict(text=f"Stations ({unit_choice})"), showticklabels=True)
    )

    pio.write_html(
        fig_scatter,
        file=scatter_path,
        include_plotlyjs='cdn',
        full_html=False
    )

    html_scatter = f"""
    <div style="height: {height-10}px; border: 1px solid #ccc; border-radius: 6px; display: flex; align-items: center; justify-content: center; overflow: hidden; margin-top: 10px;">
        <iframe loading="lazy" src="{scatter_path}" width="100%" height="100%" frameborder="0" style="flex: 3; width: 0; min-width: 0; max-width: 100%;"></iframe>
    </div>

    <div class="metric-caption">
        <strong>r²</strong> = {r2:.3f} &nbsp;|&nbsp; <strong>ME</strong> = {me:.3f} &nbsp;|&nbsp; <strong>n</strong> = {obs_vs_mod.shape[0]:.0f} (Tx NaN ≤ {missing_choice})
    </div>
    """

    mean_mod = obs_vs_mod.select(pl.col("AROME").mean()).to_numpy()
    mean_obs = obs_vs_mod.select(pl.col("Station").mean()).to_numpy()

    return html_map_legend, html_scatter, r2, me, obs_vs_mod.shape[0], me / np.mean([mean_mod, mean_obs])