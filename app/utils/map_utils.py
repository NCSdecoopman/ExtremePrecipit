import pydeck as pdk
import streamlit as st

def create_layer(df):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df.to_dict(orient="records"),
        get_position=["lon", "lat"],
        get_fill_color="fill_color",  # utilise les données RGBA fournies
        get_radius=6000,
        radius_min_pixels=1,
        radius_max_pixels=100,
        pickable=True,
        opacity=1.0
    )


def create_tooltip(stat, label):
    return {
            "html": f"""
                ({{lat_fmt}}, {{lat_fmt}})<br>
                {{val_fmt}} {label}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

def plot_map(layer, view_state, tooltip):
    # Création du Deck avec toutes les couches
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light"  
    )
    return deck
