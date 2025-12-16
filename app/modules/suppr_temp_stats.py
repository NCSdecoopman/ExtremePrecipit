import streamlit as st

from app.utils.map_utils import plot_map
from app.utils.legends_utils import get_stat_unit

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map

import polars as pl

from app.utils.hist_utils import plot_histogramme, plot_histogramme_comparatif
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive
from app.utils.show_info import show_info_data, show_info_metric
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics

import streamlit as st
import polars as pl

from app.utils.hist_utils import plot_histogramme, plot_histogramme_comparatif
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive
from app.utils.show_info import show_info_data, show_info_metric
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics

import streamlit as st
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

from app.utils.show_info import show_info_data, show_info_metric
from app.utils.data_utils import match_and_compare
from app.utils.stats_utils import generate_metrics
from app.utils.scatter_plot_utils import generate_scatter_plot_interactive


def pipeline_scatter(params_load):
    (
        df_modelised_load,
        df_observed_load,
        column_to_show,
        result_df_modelised,
        result_df_modelised_show,
        result_df_observed,
        stat_choice_key,
        scale_choice_key,
        stat_choice,
        unit_label,
        height
    ) = params_load

    # Informations globales
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7)
    n_tot_mod = df_modelised_load.select(pl.col("NUM_POSTE").n_unique()).item()
    n_tot_obs = df_observed_load.select(pl.col("NUM_POSTE").n_unique()).item()
    show_info_data(col0, "CP-AROME map", result_df_modelised_show.shape[0], n_tot_mod)
    show_info_data(col1, "Stations", result_df_observed.shape[0], n_tot_obs)

    if stat_choice_key in ["date", "month"]:
        plot_histogramme_comparatif(
            result_df_observed,
            result_df_modelised,
            column_to_show,
            stat_choice,
            stat_choice_key,
            unit_label,
            height
        )
        return None

    # Lecture de la correspondance station -> grille
    echelle = "horaire" if scale_choice_key == "mm_h" else "quotidien"
    df_map = (
        pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle}.csv")
        .with_columns([
            pl.col("NUM_POSTE_obs").cast(pl.Utf8),
            pl.col("NUM_POSTE_mod").cast(pl.Utf8)
        ])
        .select(["NUM_POSTE_obs", "NUM_POSTE_mod", "lat_mod", "lon_mod"]).unique()
    )
    df_obs = result_df_observed.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))
    df_mod = result_df_modelised.with_columns(pl.col("NUM_POSTE").cast(pl.Utf8))

    # 1) Scatter brut : chaque station vs grille
    obs_vs_mod = match_and_compare(df_obs, df_mod, column_to_show, df_map)
    if obs_vs_mod is None or obs_vs_mod.height == 0:
        st.warning("Aucune correspondance obs vs mod trouvée.")
        return None
    
    colscatt1, colscatt2 = st.columns(2)

    with colscatt1:
        # Affichage du scatter brut
        me, mae, rmse, r2 = generate_metrics(obs_vs_mod)
        st.markdown("#### Scatter plot : chaque station vs grille AROME")
        fig1 = generate_scatter_plot_interactive(obs_vs_mod, stat_choice, unit_label, height)
        st.plotly_chart(fig1, use_container_width=True)
        show_info_metric(col2, "ME", me)
        show_info_metric(col3, "MAE", mae)
        show_info_metric(col4, "RMSE", rmse)
        show_info_metric(col5, "r²", r2)

    # 2) Scatter moyen : moyenne des obs vs moyenne des modélisations par grille
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import haversine_distances

    # Préparation des coordonnées des observations (radians pour haversine)
    coords_obs = (
        df_obs.select(["lat", "lon"])
        .unique()
        .with_columns([
            (pl.col("lat") * np.pi / 180).alias("lat_rad"),
            (pl.col("lon") * np.pi / 180).alias("lon_rad")
        ])
    )

    coords_rad = coords_obs.select(["lat_rad", "lon_rad"]).to_numpy()
    if coords_rad.shape[0] < 2:
        st.warning("Pas assez de points pour regrouper spatialement.")
        return r2

    # Clustering spatial avec rayon de 5 km (~0.0784 radians)
    kms_per_radian = 6371.0088
    radius_km = 2.5
    epsilon = radius_km / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    coords_obs = coords_obs.with_columns(pl.Series(name="cluster_id", values=db.labels_))

    # Attribution du cluster aux observations
    df_obs_with_cluster = (
        df_obs.join(coords_obs.select(["lat", "lon", "cluster_id"]), on=["lat", "lon"], how="left")
    )

    # Moyenne des observations par cluster
    obs_clustered = (
        df_obs_with_cluster
        .group_by("cluster_id")
        .agg(pl.col(column_to_show).mean().alias("mean_obs"))
    )

    # Moyenne des coordonnées de chaque cluster pour localisation
    cluster_coords = (
        df_obs_with_cluster
        .group_by("cluster_id")
        .agg([
            pl.col("lat").mean().alias("lat_cluster"),
            pl.col("lon").mean().alias("lon_cluster")
        ])
    )

    # Jointure avec les coordonnées moyennes
    obs_clustered = obs_clustered.join(cluster_coords, on="cluster_id", how="left")

    # Pour chaque cluster, trouver les modélisations dans un rayon de 5 km et moyenner
    mod_points = df_mod.select(["lat", "lon", "NUM_POSTE", column_to_show]).unique()

    mod_means = []
    for row in obs_clustered.iter_rows(named=True):
        lat0, lon0 = row["lat_cluster"], row["lon_cluster"]
        lat0_rad, lon0_rad = np.radians([lat0, lon0])
        dists = haversine_distances(
            np.radians(mod_points.select(["lat", "lon"]).to_numpy()),
            np.array([[lat0_rad, lon0_rad]])
        ) * kms_per_radian
        mask = dists[:, 0] <= 5
        nearby_mods = mod_points.filter(mask)
        if nearby_mods.height > 0:
            mean_mod = nearby_mods[column_to_show].mean()
            mod_means.append(mean_mod)
        else:
            mod_means.append(None)

    # Ajouter les moyennes modélisées aux clusters
    obs_clustered = obs_clustered.with_columns(pl.Series(name="mean_mod", values=mod_means))

    # Nettoyage : supprimer les lignes avec valeurs manquantes
    clustered_final = obs_clustered.drop_nulls(["mean_obs", "mean_mod"])
    if clustered_final.height == 0:
        st.warning("Aucune correspondance moyennée trouvée.")
        return r2

    # Renommage et scatter plot
    scatter_df = clustered_final.rename({"mean_obs": "pr_obs", "mean_mod": "pr_mod"})
    me3, mae3, rmse3, r2_3 = generate_metrics(scatter_df)

    scatter_df = (
        clustered_final
        .rename({"mean_obs": "pr_obs", "mean_mod": "pr_mod"})
        .with_columns([
            # on replace lat_cluster → lat, lon_cluster → lon
            pl.col("lat_cluster").alias("lat"),
            pl.col("lon_cluster").alias("lon"),
            # colonnes vides pour satisfaire le select interne
            pl.lit("").alias("NUM_POSTE_obs"),
            pl.lit("").alias("NUM_POSTE_mod"),
        ])
    )

    with colscatt2:
        st.markdown(f"#### Scatter plot : observations et modélisations moyennées par groupe spatial (rayon {radius_km} km)")
        show_info_data(col1, "Stations", scatter_df.height, n_tot_obs)
        fig3 = generate_scatter_plot_interactive(scatter_df, stat_choice, unit_label, height)
        st.plotly_chart(fig3, use_container_width=True)
        show_info_metric(col2, f"ME ({radius_km} km)", me3)
        show_info_metric(col3, f"MAE ({radius_km} km)", mae3)
        show_info_metric(col4, f"RMSE ({radius_km} km)", rmse3)
        show_info_metric(col5, f"r² ({radius_km} km)", r2_3)



    # 1) Caster NUM_POSTE_obs en chaîne de caractères dans obs_vs_mod
    obs_vs_mod_str = obs_vs_mod.with_columns(
        pl.col("NUM_POSTE_obs").cast(pl.Utf8)
    )

    # 2) Préparer la table de clusters en s’assurant que NUM_POSTE_obs est aussi Utf8
    df_obs_clusters = (
        df_obs_with_cluster
        .select([
            pl.col("NUM_POSTE").cast(pl.Utf8).alias("NUM_POSTE_obs"),
            "cluster_id"
        ])
    )

    # 3) On peut maintenant joindre sans erreur de type
    obs_vs_mod_clustered = obs_vs_mod_str.join(
        df_obs_clusters,
        on="NUM_POSTE_obs",
        how="left"
    )

    # 4) Et filtrer les clusters à ≥ 2 stations
    obs_vs_mod_filtered = obs_vs_mod_clustered.filter(
        pl.count().over("cluster_id") >= 2
    )

    # 1. Récupère les clusters valides
    valid_clusters = (
        df_obs_with_cluster
        .group_by("cluster_id")
        .agg(pl.count().alias("n_stations"))
        .filter(pl.col("n_stations") >= 2)
        .select("cluster_id")
    )

    # 2. Rejoins avec clustered_final pour filtrer les clusters valides
    scatter_df_filtered = (
        clustered_final
        .join(valid_clusters, on="cluster_id", how="inner")
        .rename({"mean_obs": "pr_obs", "mean_mod": "pr_mod"})
        .with_columns([
            pl.col("lat_cluster").alias("lat"),
            pl.col("lon_cluster").alias("lon"),
            pl.lit("").alias("NUM_POSTE_obs"),
            pl.lit("").alias("NUM_POSTE_mod"),
        ])
    )


    with colscatt1:
        st.markdown(f"#### Scatter plot : observations et modélisations qui seront clustérisés")
        show_info_data(col1, "Stations", obs_vs_mod_filtered.height, n_tot_obs)
        fig4 = generate_scatter_plot_interactive(obs_vs_mod_filtered, stat_choice, unit_label, height)
        st.plotly_chart(fig4, use_container_width=True)
        me3, mae3, rmse3, r2_3 = generate_metrics(obs_vs_mod_filtered)
        show_info_metric(col2, f"ME", me3)
        show_info_metric(col3, f"MAE", mae3)
        show_info_metric(col4, f"RMSE", rmse3)
        show_info_metric(col5, f"r²", r2_3)

    with colscatt2:
        st.markdown(f"#### Scatter plot : observations et modélisations moyennées par groupe spatial (rayon {radius_km} km)")
        show_info_data(col1, "Stations", scatter_df_filtered.height, n_tot_obs)
        fig5 = generate_scatter_plot_interactive(scatter_df_filtered, stat_choice, unit_label, height)
        st.plotly_chart(fig5, use_container_width=True)
        me3, mae3, rmse3, r2_3 = generate_metrics(scatter_df_filtered)
        show_info_metric(col2, f"ME ({radius_km} km)", me3)
        show_info_metric(col3, f"MAE ({radius_km} km)", mae3)
        show_info_metric(col4, f"RMSE ({radius_km} km)", rmse3)
        show_info_metric(col5, f"r² ({radius_km} km)", r2_3)

    return r2





def show(config_path):
    st.markdown("<h3>Visualisation des précipitations</h3>", unsafe_allow_html=True)
    
    # Chargement des config
    params_config = pipeline_config(config_path)
    config = params_config["config"]
    stat_choice = params_config["stat_choice"]
    season_choice = params_config["season_choice"]
    stat_choice_key = params_config["stat_choice_key"]
    scale_choice_key = params_config["scale_choice_key"]
    min_year_choice = params_config["min_year_choice"]
    max_year_choice = params_config["max_year_choice"]
    season_choice_key = params_config["season_choice_key"]
    missing_rate = params_config["missing_rate"]
    quantile_choice = params_config["quantile_choice"]

    # Préparation des paramètres pour pipeline_data
    params_load = (
        stat_choice_key,
        scale_choice_key,
        min_year_choice,
        max_year_choice,
        season_choice_key,
        missing_rate,
        quantile_choice
    )

    # Chargement des données
    params_load = (
        stat_choice_key,
        scale_choice_key,
        min_year_choice,
        max_year_choice,
        season_choice_key,
        missing_rate,
        quantile_choice
    )
    result = pipeline_data(params_load, config)
    df_modelised_load = result["modelised_load"]
    df_observed_load = result["observed_load"]
    result_df_modelised_show = result["modelised_show"]
    result_df_modelised = result["modelised"]
    result_df_observed = result["observed"]
    column_to_show = result["column"]

    # Chargement des affichages graphiques
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    height=650
    params_map = (
        stat_choice_key,
        column_to_show,
        result_df_modelised_show,
        result_df_observed,
        unit_label,
        height
    )
    layer, scatter_layer, tooltip, view_state, html_legend = pipeline_map(params_map)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        st.markdown(
            f"""
            <div style='text-align: left; margin-bottom: 10px;'>
                <b>{stat_choice} des précipitations de {min_year_choice} à {max_year_choice} ({season_choice.lower()})</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        if deck:
            st.pydeck_chart(deck, use_container_width=True, height=height)
        st.markdown(
            """
            <div style='text-align: left; font-size: 0.8em; color: grey; margin-top: 0px;'>
                Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(html_legend, unsafe_allow_html=True)        


    params_scatter = (
        df_modelised_load, 
        df_observed_load, 
        column_to_show, 
        result_df_modelised, 
        result_df_modelised_show, 
        result_df_observed, 
        stat_choice_key, 
        scale_choice_key, 
        stat_choice,unit_label, 
        height
    )
    pipeline_scatter(params_scatter)