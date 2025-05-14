from app.utils.config_utils import echelle_config
from app.utils.map_utils import create_layer, create_scatter_layer, create_tooltip
from app.utils.legends_utils import formalised_legend, display_vertical_color_legend

import pydeck as pdk

def safe_min(*args):
    return min(x for x in args if x is not None) if any(x is not None for x in args) else None

def safe_max(*args):
    return max(x for x in args if x is not None) if any(x is not None for x in args) else None


def pipeline_map(
    params_load, 
    n_colors:int = 15,
    param_view: dict = {"latitude": 46.9, "longitude": 1.7, "zoom": 5}
):
    # Déballage des paramètres
    stat_choice_key, result, unit_label, height = params_load

    # Echelle continue ou discrète
    if "continu" in result:
        continu = result["continu"]
    elif stat_choice_key == "month":
        continu = False
    else:
        continu = True

    # Nombre de couleurs
    if "categories" in result: # Discret
        categories = result["categories"]
        n_colors = len(categories)
    else:
        categories = None
        n_colors = n_colors

    # Echelle paramétrée par l'utilisateur
    if "echelle" not in result: # Choix d'une échelle personnalisée
        result["echelle"] = None

    # On trouve alors la représéntation de la légende
    colormap = echelle_config(continu, echelle=result["echelle"], n_colors=n_colors)

    result_df_modelised_show = result["modelised_show"]
    result_df_observed_show = result["observed_show"]
   
    # Normalisation des valeurs modélisées
    result_df_modelised_show, vmin_mod, vmax_mod = formalised_legend(
        result["modelised_show"], 
        column_to_show=result["column"], 
        colormap=colormap,
        is_categorical=not continu,
        categories=categories
    )

    # Normalisation des observations avec les mêmes bornes
    result_df_observed_show, vmin_obs, vmax_obs = formalised_legend(
        result["observed_show"], 
        column_to_show=result["column"], 
        colormap=colormap,
        is_categorical=not continu,
        categories=categories
    )


    # Calcul des bornes communes
    if "vmin" in result and "vmax" in result:
        vmin_commun, vmax_commun = result["vmin"], result["vmax"]
    else:
        vmin_commun = safe_min(vmin_mod, vmin_obs)
        vmax_commun = safe_max(vmax_mod, vmax_obs)

    # Mise à jour de la normalisation pour les deux ensembles de données avec les bornes communes
    result_df_modelised_show, _, _ = formalised_legend(
        result["modelised_show"], 
        column_to_show=result["column"], 
        colormap=colormap,
        vmin=vmin_commun,
        vmax=vmax_commun,
        is_categorical=not continu,
        categories=categories
    )

    result_df_observed_show, _, _ = formalised_legend(
        result["observed_show"], 
        column_to_show=result["column"], 
        colormap=colormap,
        vmin=vmin_commun,
        vmax=vmax_commun,
        is_categorical=not continu,
        categories=categories
    )

    # Création du layer modélisé et observé
    layer = create_layer(result_df_modelised_show)
    scatter_layer = create_scatter_layer(result_df_observed_show)

    # Tooltip
    tooltip = create_tooltip(unit_label)

    # View par défaut
    view_state = pdk.ViewState(
        latitude=param_view["latitude"],
        longitude=param_view["longitude"], 
        zoom=param_view["zoom"]
    )

    # Légende vertical
    legend = display_vertical_color_legend(
        height, 
        colormap, 
        vmin_commun, 
        vmax_commun, 
        n_ticks=n_colors, 
        label=unit_label,
        model_labels=categories
    )

    return layer, scatter_layer, tooltip, view_state, legend