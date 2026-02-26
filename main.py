import streamlit as st
from app.modules import statisticals, gev, suppr_sauv_gev_precedent

st.set_page_config(layout="wide", page_title="Visualisation des précipitations", page_icon="🌧️")

st.markdown("""
    <style>
        body {
            font-family: 'Times New Roman', sans-serif !important;
            font-size: 10px !important;
            font-weight: 400 !important;
        }            
        html, body {
            font-size: 10px !important;
        }

        * {
            font-size: 10px !important;
        }
            
        h1, h2, h3, h4, h5, h6, label, button {
            font-size: 10px !important;
        }
            
        section[data-testid="stSidebar"] {
            width: 150px !important;
            min-width: 150px !important;
        }  
                 
        /* Hide app header */
        div[class*="stAppHeader"] {
            display: none !important;
        }
            
        div[class*="block-container"] {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 98% !important;
        }

        /* Responsive column layout */
        @media screen and (max-width: 1000px) {
            .element-container:has(> .stColumn) {
                display: flex;
                flex-wrap: wrap;
            }

            .element-container:has(> .stColumn) .stColumn {
                width: 48% !important;
                min-width: 48% !important;
            }
        }

        @media screen and (max-width: 600px) {
            .element-container:has(> .stColumn) .stColumn {
                width: 100% !important;
                min-width: 100% !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "Navigation",
    ("Choisir un visuel", "Statistiques descriptives", "Variation décennale"),
    index=0  # Default to first option
)

config_path = "app/config/config.yaml"

if option == "Statistiques descriptives":
    statisticals.show(config_path)

elif option == "Variation décennale":
    gev.show(config_path)

elif option == "temp":
    suppr_sauv_gev_precedent.show(config_path)

else:
    st.markdown("""
    <div style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 20px;">
    Analyse interactive des précipitations en France (1959–2022)
    </div>

    ### 🚀 Objectif de l’application
    Cette application a pour mission de **mettre en lumière** et **d’évaluer** les tendances des précipitations extrêmes, à la fois **journalières** et **horaires**, sur tout le territoire français. Pour ce faire, nous nous appuyons sur :  
    - Le modèle régional **CP-RCM CNRM-AROME** (résolution 2,5 km) forcé par les réanalyses **ERA5**.  
    - La riche base d’observations de **Météo-France** pour valider et comparer les résultats.

    ### 🔍 Ce que vous pouvez faire
    1. **Explorer des statistiques descriptives**  
    • Carte interactive des intensités moyennes et maximales (journalières et horaires)  
    • Historiques et distributions par région  
    2. **Comparer modèle vs. observations**  
    • Visualisation des écarts entre les sorties modélisées et les relevés Météo-France  
    • Évaluation de la fiabilité du modèle à haute résolution  
    3. **Analyser l’impact du changement climatique**  
    • Suivi des évolutions spatiales et temporelles des extrêmes  
    • Mise en évidence des régions sensibles

    ### 🌍 Pourquoi c’est important
    - 🔸 **Comprendre l’évolution** des pluies intenses pour mieux prévenir les risques d’inondation.  
    - 🔸 **Mettre en perspective** l’effet du réchauffement climatique sur la variabilité pluviométrique.  
    - 🔸 **Fournir un outil interactif** aux chercheurs, aux collectivités et aux citoyens pour explorer ces phénomènes en temps réel.


    <p style="text-align: center; font-style: italic; margin-top: 20px;">
    Plongez au cœur des données, explorez les cartes et laissez-vous guider par l’évolution des précipitations extrêmes !
    </p>
    """, unsafe_allow_html=True)