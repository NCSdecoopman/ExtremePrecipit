from app.utils.config_utils import echelle_config
from app.utils.map_utils import create_layer, create_scatter_layer, create_tooltip
from app.utils.legends_utils import formalised_legend, display_vertical_color_legend

import pydeck as pdk


def pipeline_map(params_load):
    # Déballage des paramètres
    stat_choice_key, column_to_show, result_df_modelised_show, result_df_observed, unit_label, height = params_load

    # Définir l'échelle personnalisée continue ou discrète selon le cas
    colormap = echelle_config("continu" if stat_choice_key != "month" else "discret", n_colors=15)

    # Normalisation de la légende pour les valeurs modélisées
    result_df_modelised_show, vmin, vmax = formalised_legend(result_df_modelised_show, column_to_show, colormap)

    # Création du layer modélisé
    layer = create_layer(result_df_modelised_show)

    # Normalisation des points observés avec les mêmes bornes
    result_df_observed, _, _ = formalised_legend(result_df_observed, column_to_show, colormap, vmin, vmax)
    scatter_layer = create_scatter_layer(result_df_observed, radius=1500)

    # Tooltip (tu dois t'assurer que unit_label est défini quelque part ou passé en paramètre)
    tooltip = create_tooltip(unit_label)

    # View par défaut
    view_state = pdk.ViewState(latitude=46.5, longitude=1.7, zoom=5)

    # Légende vertical
    html_legend = display_vertical_color_legend(height, colormap, vmin, vmax, n_ticks=15, label=unit_label)

    return layer, scatter_layer, tooltip, view_state, html_legend
