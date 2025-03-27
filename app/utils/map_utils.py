import pydeck as pdk
import streamlit as st

def create_layer(df):
    return pdk.Layer(
        "GridCellLayer",
        data=df,
        get_position=["lon", "lat"],
        elevation_scale=0,
        cell_size=2500,
        get_fill_color="fill_color",
        pickable=True
    )

def create_tooltip(stat):
    return {
            "html": f"""
                <b>{stat}</b>: {{val_fmt}}<br>
                <b>Lat</b>: {{lat}}<br>
                <b>Lon</b>: {{lon}}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

def plot_map(layer, view_state, tooltip):
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        height=600
    )

    st.pydeck_chart(deck, use_container_width=True)