import streamlit as st
from streamlit_folium import st_folium
import folium

from pathlib import Path
from functools import reduce

from app.utils.config_utils import *
from app.utils.menus_utils import *
from app.utils.data_utils import *
from app.utils.stats_utils import *
from app.utils.map_utils import *
from app.utils.legends_utils import *
from app.utils.hist_utils import *
from app.utils.scatter_plot_utils import *

import pydeck as pdk
import polars as pl

def filter_nan(df: pl.DataFrame):
    return df.drop_nulls(subset=["xi", "mu", "sigma"])

def filter_percentile_all(df: pl.DataFrame, quantile_choice: float, columns: list[str]):
    from functools import reduce
    quantiles = {
        col: df.select(pl.col(col).quantile(quantile_choice, "nearest")).item()
        for col in columns
    }
    filter_expr = reduce(
        lambda acc, col: acc & (pl.col(col) <= quantiles[col]),
        columns[1:],
        pl.col(columns[0]) <= quantiles[columns[0]]
    )
    return df.filter(filter_expr)

def show(config_path):
    st.markdown("<h3>Visualisation des paramètres GEV</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        Echelle = st.selectbox("Choix de l'échelle temporelle", ["Horaire", "Journalière"], key="scale_choice")
    with col2:
        quantile_choice = st.slider(
            "Percentile de retrait",
            min_value=0.950,
            max_value=1.00,
            value=0.999,
            step=0.001,
            format="%.3f",
            key="quantile_choice"
        )

    echelle = Echelle.lower()
    echelle = "quotidien" if echelle == "journalière" else echelle

    config = load_config(config_path)
    mod_dir = Path(config["gev"]["modelised"]) / echelle
    obs_dir = Path(config["gev"]["observed"]) / echelle

    try:
        df_modelised = pl.read_parquet(mod_dir / "gev_param.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres modélisés : {e}")
        return

    try:
        df_observed = pl.read_parquet(obs_dir / "gev_param.parquet")
    except Exception as e:
        st.error(f"Erreur lors du chargement des paramètres observés : {e}")
        return
    
    # Suppression des NaN
    df_observed = filter_nan(df_observed)

    df_modelised = add_alti(df_modelised, type='model')
    df_observed = add_alti(df_observed, type=echelle)

    params_gev = {
        "xi": "ξ",
        "mu": "μ",
        "sigma": "σ"
    }

    df_modelised_show = filter_percentile_all(df_modelised, quantile_choice, ["xi", "mu", "sigma"])

    colormap = echelle_config("continu", n_colors=15)
    view_state = pdk.ViewState(latitude=46.9, longitude=1.7, zoom=4.6)
    tooltip = create_tooltip("")

    col1, col2, col3 = st.columns(3)
    for param, label in params_gev.items():
        vmin_modelised = df_modelised_show[param].min()
        df_modelised_leg, vmin, vmax = formalised_legend(df_modelised_show, param, colormap, vmin=vmin_modelised)
        df_observed_leg, _, _ = formalised_legend(df_observed, param, colormap, vmin, vmax)

        layer = create_layer(df_modelised_leg)
        scatter_layer = create_scatter_layer(df_observed_leg, radius=1500)

        deck = plot_map([layer, scatter_layer], view_state, tooltip)
        with eval(f"col{list(params_gev).index(param)+1}"):
            cola, colb = st.columns([0.9, 0.1])
            with cola:
                st.markdown(f"<b>Visualisation de {label}</b>", unsafe_allow_html=True)
                st.pydeck_chart(deck, use_container_width=True, height=450)
            with colb:
                display_vertical_color_legend(500, colormap, vmin, vmax, n_ticks=15, label=label)
    

    # Partie intéractive
    df_files = list_files_and_metadata_hf(echelle)

    # Carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=6, tiles="OpenStreetMap")

    # Palette bleu → blanc → rouge centrée sur 0
    vmin = df_files["me"].min()
    vmax = df_files["me"].max()
    vcenter = 0  # centre sur 0

    # Colormap personnalisée avec blanc au centre
    colorscale = LinearSegmentedColormap.from_list(
        "blue_white_red", ["blue", "white", "red"], N=256
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Création carte Folium
    m = folium.Map(location=[46.9, 1.7], zoom_start=6, tiles="OpenStreetMap")

    # Ajout des points
    for _, row in df_files.iterrows():
        me = row["me"]
        rgba = colorscale(norm(me))
        hex_color = colors.rgb2hex(rgba)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            tooltip=f"lat: {row['lat']:.2f}, lon: {row['lon']:.2f}, me: {row['me']:.2f}"
        ).add_to(m)


    # Ajout d’une échelle personnalisée (branca)
    colormap = LinearColormap(
        colors=["blue", "white", "red"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "Erreur moyenne (ME)"
    colormap.add_to(m)

    col1, col2 = st.columns([2, 3.5])

    with col1:
        # Capture du clic utilisateur
        m.add_child(folium.LatLngPopup())
        map_output = st_folium(m, width=600, height=600)
        # Traitement du clic
        if map_output and map_output.get("last_clicked"):
            lat_clicked = map_output["last_clicked"]["lat"]
            lon_clicked = map_output["last_clicked"]["lng"]

            st.success(f"Point sélectionné : lat = {lat_clicked:.4f}, lon = {lon_clicked:.4f}")
            # Trouver le fichier le plus proche (tolérance arrondi 3 décimales)
            row = df_files[
                (df_files["lat"] == lat_clicked) & (df_files["lon"] == lon_clicked)
            ]
        else:
            row = None

    with col2:
        retrait_0_0 = st.checkbox("Retirer les points où AROME = 0 et OBS = 0 pour une même date (pas de pluie)", value=True)            
        if row is not None and not row.empty:
            file = row.iloc[0]["file"]
            df = import_data_hf(file)

            if retrait_0_0:
                df = df[~((df["pr_mod"] == 0) & (df["pr_obs"] == 0))]
            
            df = pd.DataFrame({
                "arome": df["pr_mod"],
                "obs": df["pr_obs"],
                "Date": df["time"]
            })

            # Calcul des métriques
            rmse = np.sqrt(mean_squared_error(df["obs"], df["arome"]))
            mae = mean_absolute_error(df["obs"], df["arome"])
            bias = np.mean(df["arome"] - df["obs"])
            r2 = r2_score(df["obs"], df["arome"])

            # Affichage
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            with col_metrics1:
                st.metric("RMSE", f"{rmse:.2f}")
            with col_metrics2:
                st.metric("MAE", f"{mae:.2f}")
            with col_metrics3:
                st.metric("Biais", f"{bias:.2f}")
            with col_metrics4:
                st.metric("R²", f"{r2:.3f}")

            col3, col4, col5 = st.columns([2.5, 2.5, 2.5])

            with col3:
                # Exemple d'appel Streamlit
                fig = generate_scatter_plot_interactive(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                # Ou :
                fig = generate_hexbin_plot_interactive(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            with col5:
                fig = generate_error_histogram(df, "arome", "obs")
                st.plotly_chart(fig, use_container_width=True)

            
