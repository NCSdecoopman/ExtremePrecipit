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
                ({{lat}}, {{lon}})<br>
                {{val_fmt}}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

def plot_map(layer, view_state, tooltip):
    # # Charger le GeoJSON des départements (directement depuis l'URL)
    # url_geojson = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    # response = requests.get(url_geojson)
    # geojson = response.json()

    # # Couche GeoJsonLayer : contours des départements
    # geojson_layer = pdk.Layer(
    #     "GeoJsonLayer",
    #     data=geojson,
    #     pickable=True,
    #     stroked=True,
    #     filled=True,
    #     extruded=False,
    #     get_fill_color=[220, 220, 220],
    #     get_line_color=[100, 100, 100],
    # )

    # # Calcul des centroïdes pour placer les noms de départements
    # noms, lons, lats = [], [], []
    # for feature in geojson["features"]:
    #     nom = feature["properties"]["nom"]
    #     geom = shape(feature["geometry"])
    #     centroid = geom.centroid
    #     noms.append(nom)
    #     lons.append(centroid.x)
    #     lats.append(centroid.y)

    # df_text = pd.DataFrame({
    #     "nom": noms,
    #     "lon": lons,
    #     "lat": lats
    # })

    # # Couche TextLayer : noms des départements
    # text_layer = pdk.Layer(
    #     "TextLayer",
    #     data=df_text,
    #     get_position=["lon", "lat"],
    #     get_text="nom",
    #     get_color=[0, 0, 0],
    #     get_size=16,
    #     get_alignment_baseline="'bottom'",
    #     pickable=False
    # )

    # Création du Deck avec toutes les couches
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    return deck
