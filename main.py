import streamlit as st
st.set_page_config(layout="wide", page_title="Visualisation des précipitations", page_icon="🌧️")

st.markdown("""
    <style>            
        /* Masquer le header de l'app */
        div[class*="stAppHeader"] {
            display: none !important;
        }
            
        div[class*="block-container"] {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 98% !important;
        }

        /* Responsive layout des colonnes */
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

from huggingface_hub import snapshot_download
import os


DATA_DIR = "data"

# Ne télécharge que si les données ne sont pas encore là
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print("📥 Téléchargement des données depuis Hugging Face Hub...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        local_dir=DATA_DIR,
        revision="main"
    )


from app.modules import statisticals

option = "Statistiques descriptives"

config_path = "app/config/config.yaml"

if option == "Statistiques descriptives":
    statisticals.show(config_path)
