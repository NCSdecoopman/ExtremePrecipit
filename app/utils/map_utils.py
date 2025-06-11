from pathlib import Path
import pydeck as pdk
import streamlit as st
import polars as pl
import geopandas as gpd

def prepare_layer(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["lat", "lon", "lat_fmt", "lon_fmt", "altitude", "val_fmt", "fill_color"]
    
    if "is_significant" in df.columns:
        cols.append("is_significant")
    
    return df.select(cols)


def fast_to_dicts(df: pl.DataFrame) -> list[dict]:
    cols = df.columns
    result = []

    # Conversion explicite des colonnes en listes Python natives
    arrays = {
        col: (
            df[col].to_list()  # pour List ou String ou autre
            if df[col].dtype == pl.List
            else df[col].to_numpy().tolist()
        )
        for col in cols
    }

    n = len(df)
    for i in range(n):
        row = {col: arrays[col][i] for col in cols}
        result.append(row)

    return result

def create_layer(df: pl.DataFrame) -> pdk.Layer:
    layers = []

    df = prepare_layer(df)

    if "is_significant" in df.columns:
        df_sig = df.filter(pl.col("is_significant"))
        df_non_sig = df.filter(~pl.col("is_significant"))
    else:
        df_sig = pl.DataFrame()
        df_non_sig = df

    # Points significatifs 
    if len(df_sig) > 0:
        df_sig = df_sig.with_columns(pl.lit("*").alias("star_text"))


        layers.append(
            pdk.Layer(
                "TextLayer",
                data=fast_to_dicts(df_sig),
                get_position=["lon", "lat"],
                get_text="star_text",
                get_size=5,
                get_color=[0, 0, 0, 255],  # noir
                get_angle=0,
                get_text_anchor="center",
                get_alignment_baseline="bottom",
                pickable=False
            )
        )

        layers.append(
            pdk.Layer(
                "GridCellLayer",
                data=fast_to_dicts(df_sig),
                get_position=["lon", "lat"],
                get_fill_color="fill_color",
                cell_size=2500,
                elevation=0,
                elevation_scale=0,
                lighting=None,
                pickable=True,
                opacity=0.2,
                extruded=False
            )
        )

    # Points non significatifs
    if len(df_non_sig) > 0:
        layers.append(
            pdk.Layer(
                "GridCellLayer",
                data=fast_to_dicts(df_non_sig), 
                get_position=["lon", "lat"],
                get_fill_color="fill_color",
                cell_size=2500,
                elevation=0,
                elevation_scale=0,
                lighting=None,
                pickable=True,
                opacity=0.2,
                extruded=False
            )
        )

    return layers


def create_scatter_layer(df: pl.DataFrame, radius=1500) -> list[pdk.Layer]:
    layers = []

    df = prepare_layer(df)

    if "is_significant" in df.columns:
        df_sig = df.filter(pl.col("is_significant"))
        df_non_sig = df.filter(~pl.col("is_significant"))
    else:
        df_sig = pl.DataFrame()
        df_non_sig = df

    # Points significatifs avec IconLayer (Triangle non rempli)
    if len(df_sig) > 0:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=fast_to_dicts(df_sig),
                get_position=["lon", "lat"],
                get_fill_color="fill_color",
                get_line_color=[0, 0, 0],
                line_width_min_pixels=0.2,
                get_radius=radius,
                radius_scale=3,
                radius_min_pixels=2,
                pickable=True,
                stroked=False
            )
        )
        

    # Points non significatifs en ScatterplotLayer classique
    if len(df_non_sig) > 0:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=fast_to_dicts(df_non_sig),
                get_position=["lon", "lat"],
                get_fill_color="fill_color",
                get_line_color=[0, 0, 0],
                line_width_min_pixels=0.2,
                get_radius=radius,
                radius_scale=1,
                radius_min_pixels=2,
                pickable=True,
                stroked=False
            )
        )

    return layers



def create_tooltip(label: str) -> dict:
    return {
        "html": f"""
            ({{lat_fmt}}, {{lon_fmt}})<br>
            {{altitude}} m<br>
            {{val_fmt}} {label}
        """,
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        },
        "condition": "altitude !== 'undefined'"
    }


def relief():
    # Lire et reprojeter le shapefile
    gdf = gpd.read_file(Path("data/external/niveaux/selection_courbes_niveau_france.shp").resolve()).to_crs(epsg=4326)

    # Extraire les chemins
    path_data = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        altitude = row["coordonnees"]  # ou la colonne correcte (parfois 'ALTITUDE', à adapter)

        if geom.geom_type == "LineString":
            path_data.append({"path": list(geom.coords), "altitude": altitude})
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                path_data.append({"path": list(line.coords), "altitude": altitude})

    # Couleur fixe noire ; possibilité d’ajouter une couleur par altitude (cf. remarque ci-dessous)
    return pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color="[0, 0, 0, 100]",  # ou map dynamique : 'd.altitude > 1000 ? [0,0,0,255] : [150,150,150,150]'
        width_scale=1,
        width_min_pixels=0.5,
        pickable=False
    )

def plot_map(layers, view_state, tooltip, activate_relief: bool=False):
    if not isinstance(layers, list):
        layers = [layers]
    
    # Supprime les couches nulles/indéfinies
    layers = [layer for layer in layers if layer is not None]

    if activate_relief:
        relief_layer = relief() 
        if relief_layer is not None:
            layers.append(relief_layer)

    try:
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=None
        )
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte : {e}")
        return None

