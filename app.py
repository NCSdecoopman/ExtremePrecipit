import os

import streamlit as st


from app.utils.data import get_available_years

from app.modules import series#, gev_stationnaire

import pandas as pd
import plotly.express as px

# from app.utils.login import login
# # Vérifie l'authentification avant d'afficher le reste de l'application
# if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#     login()
#     st.stop()  # Stoppe l'exécution si l'utilisateur n'est pas connecté

# Menu latéral
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Données à analyser :", 
    ["Test", "Statistiques descriptives"]#, "GEV stationnaire"] #, "Périodes de retours"]
)

# Définition des répertoires
OUTPUT_DIR = os.path.join("data", "result")

years = get_available_years(OUTPUT_DIR)

if not years:
    st.error("Aucune donnée disponible. Vérifie le dossier de sortie.")
    st.stop()

if option == "Statistiques descriptives":
    series.show(OUTPUT_DIR, years)

elif option =="Test":
    # DataFrame de test
    df_test = pd.DataFrame({
        'lat': [46.5, 46.6, 46.7],
        'lon': [2.0, 2.2, 2.4],
        'pr':  [1, 2, 3]
    })

    fig_test = px.scatter_mapbox(
        df_test,
        lat="lat",
        lon="lon",
        color="pr",
        color_continuous_scale=px.colors.sequential.Viridis,  # Couleurs standard
        zoom=4.5,
        center={"lat": 46.6, "lon": 2.2},
        height=500,
        title="Carte de test"
    )
    fig_test.update_layout(
        mapbox_style="open-street-map",  # Style sans token
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_test, use_container_width=True)

# elif option == "GEV stationnaire":
#     gev_stationnaire.show(OUTPUT_DIR, years)

# elif option == "Périodes de retours":
#     data = pd.read_parquet(os.path.join(OUTPUT_DIR, "gev", "quantiles_grid.parquet"))

#     option = st.sidebar.radio("Sélectionnez une option :", [
#         "Avec outliers",
#         "Sans outliers"
#     ])

#     # Définition des périodes de retour
#     return_periods = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#     # Suppression des outliers si sélectionné
#     if option == "Sans outliers":
#         def remove_outliers_gev(df, columns):
#             mu = df[columns].mean()
#             sigma = df[columns].std()
#             lower_bound = mu - 3 * sigma
#             upper_bound = mu + 3 * sigma
#             return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]
        
#         data = remove_outliers_gev(data, [str(T) for T in return_periods])

#     # Création des onglets
#     tabs_periode = st.tabs(["Carte des quantiles", "Quantiles moyens"])

#     with tabs_periode[0]:
#         # Sélection d'une seule période pour la carte
#         selected_period = st.selectbox("Sélectionnez une période de retour :", return_periods, index=2)
        
#         # Création de la carte
#         fig_map = px.scatter_mapbox(
#             data,
#             lat="lat",
#             lon="lon",
#             color=data[str(selected_period)],
#             color_continuous_scale="viridis",
#             size_max=1,
#             zoom=4.5,
#             mapbox_style="carto-darkmatter",
#             title=f"Quantiles pour une période de retour de {selected_period} ans",
#             width=1000,
#             height=700
#         )

#         fig_map.update_layout(
#             coloraxis_colorbar=dict(
#                 title="mm/h"
#             ),
#             dragmode=False
#         )

#         st.plotly_chart(fig_map, use_container_width=True)

#         # Affichage des statistiques descriptives
#         stats_df = pd.DataFrame({
#             "Statistique": ["Moyenne", "Médiane", "Minimum", "Maximum", "Écart-type"],
#             "Valeur": [
#                 data[str(selected_period)].mean(),
#                 data[str(selected_period)].median(),
#                 data[str(selected_period)].min(),
#                 data[str(selected_period)].max(),
#                 data[str(selected_period)].std()
#             ]
#         })
#         stats_df["Valeur"] = stats_df["Valeur"].round(2)
#         st.dataframe(stats_df, hide_index=True)

#     with tabs_periode[1]:
#         import plotly.graph_objects as go
#         # Transformation des données pour le graphique
#         melted_data = data.melt(id_vars=["lat", "lon"], value_vars=[str(T) for T in return_periods],
#                                 var_name="Période de retour", value_name="Quantile")
#         melted_data["Période de retour"] = melted_data["Période de retour"].astype(int)

#         # Calcul du quantile moyen par période de retour
#         summary_stats = melted_data.groupby("Période de retour")["Quantile"].agg(["mean", "std"]).reset_index()
#         summary_stats.rename(columns={"mean": "Quantile moyen", "std": "Écart-type"}, inplace=True)

#         # Création du graphique avec bande d'incertitude
#         fig = go.Figure()

#         # Ajout de la bande d'incertitude (écart-type)
#         fig.add_trace(go.Scatter(
#             x=summary_stats["Période de retour"],
#             y=summary_stats["Quantile moyen"] + summary_stats["Écart-type"],
#             mode="lines",
#             line=dict(width=0),
#             name="Quantile moyen + Écart-type",
#             showlegend=False
#         ))

#         fig.add_trace(go.Scatter(
#             x=summary_stats["Période de retour"],
#             y=summary_stats["Quantile moyen"] - summary_stats["Écart-type"],
#             mode="lines",
#             line=dict(width=0),
#             fill='tonexty',  # Remplissage entre les deux courbes
#             fillcolor='rgba(0, 100, 250, 0.2)',  # Couleur semi-transparente
#             name="Quantile moyen - Écart-type",
#             showlegend=False
#         ))

#         # Ajout de la courbe du quantile moyen
#         fig.add_trace(go.Scatter(
#             x=summary_stats["Période de retour"],
#             y=summary_stats["Quantile moyen"],
#             mode="lines+markers",
#             name="Quantile moyen",
#             line=dict(color='blue'),
#             showlegend=False
#         ))

#         # Mise en forme du graphique
#         fig.update_layout(
#             title="Quantile moyen par période de retour",
#             xaxis_title="Période de retour (ans)",
#             yaxis_title="Quantile moyen (mm/h)",
#             width=800,
#             height=600
#         )

#         # Affichage dans Streamlit
#         st.plotly_chart(fig, use_container_width=True)
