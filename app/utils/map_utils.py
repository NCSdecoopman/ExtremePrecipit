import pydeck as pdk
import streamlit as st
import polars as pl

def prepare_layer(df: pl.DataFrame) -> pl.DataFrame:
    if "altitude" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("altitude"))

    return df.select([
        "lat", "lon", "lat_fmt", "lon_fmt", "altitude", "val_fmt", "fill_color"
    ])

def create_layer(df: pl.DataFrame) -> pdk.Layer:
    df = prepare_layer(df)

    return pdk.Layer(
        "GridCellLayer",
        data=df.to_dicts(), 
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

def create_scatter_layer(df: pl.DataFrame, radius=1000) -> pdk.Layer:
    df = prepare_layer(df)

    return pdk.Layer(
        "ScatterplotLayer",
        data=df.to_dicts(),
        get_position=["lon", "lat"],
        get_fill_color="fill_color",
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        get_radius=radius,
        radius_scale=1,
        radius_min_pixels=2,
        pickable=True,
        stroked=True
    )

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

def plot_map(layers, view_state, tooltip):
    if not isinstance(layers, list):
        layers = [layers]

    try:
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light"
        )
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte : {e}")
        return None
