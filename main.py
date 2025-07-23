import streamlit as st

from app.utils.map_utils import plot_map
from app.utils.legends_utils import get_stat_unit

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter
from app.utils.show_info import show_info_data, show_info_metric

st.set_page_config(layout="wide", page_title="Analyse interactive des précipitations en France (1959–2022)", page_icon="🌧️")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        * {
            font-size: 10px !important;
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

css = """
<style>
/* -------------------- VARIABLES GLOBALES -------------------- */
:root{
  --primary:#5A7BFF;
  --primary-light:#8FA0FF;
  --accent:#FF7A59;
  --bg:rgba(245,247,250,0.65);
  --card:rgba(255,255,255,0.35);
  --text:#1F2D3D;
  --text-light:#6B7C93;
  --radius:18px;
  --shadow:0 12px 28px rgba(0,0,0,.12);
  --blur:18px;
  --font:"Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* -------------------- RESET & BODY -------------------- */
html, body, [class*="stAppViewContainer"]{
  font-family: var(--font) !important;
  color: var(--text);
}
body{
  background: linear-gradient(135deg,#EEF2FF 0%,#FDFBFF 60%,#F0F4FF 100%) fixed !important;
}

/* Conteneur principal */
.block-container{
  padding-top: 2.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 98%;
}

/* -------------------- EN-TÊTES -------------------- */
h1,h2,h3,h4{
  font-weight: 600 !important;
  letter-spacing: -0.01em;
  color: var(--text);
}
h1{
  font-size: 2.1rem !important;
  margin-bottom: 1.2rem;
}

/* -------------------- CARTES / WIDGETS -------------------- */
section.main > div{
  backdrop-filter: blur(var(--blur));
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.5rem 1.8rem;
}

/* -------------------- LABELS DES WIDGETS -------------------- */
.css-10trblm, .stSlider label, .stSelectbox label, .stNumberInput label, .stMultiSelect label{
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  color: var(--text-light) !important;
  margin-bottom: .4rem !important;
  text-transform: uppercase;
  letter-spacing: .04em;
}

/* -------------------- SELECTBOX -------------------- */
.stSelectbox > div div[data-baseweb="select"]{
  background: rgba(255,255,255,0.55);
  border-radius: var(--radius) !important;
  border: 1px solid rgba(0,0,0,.05);
  box-shadow: inset 0 2px 4px rgba(0,0,0,.04);
}
.stSelectbox > div div[data-baseweb="select"]:hover{
  border-color: var(--primary-light);
}
.stSelectbox svg{
  stroke: var(--primary) !important;
}

/* -------------------- SLIDER -------------------- */
[data-testid="stSlider"] > div{
  padding-top: .6rem;
}
[data-testid="stSlider"] [data-testid="stThumbValue"]{
  background: var(--primary);
  color: #fff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: .75rem;
  box-shadow: var(--shadow);
}
[data-testid="stSlider"] [data-testid="stTickBar"]{
  background: rgba(0,0,0,.08);
}
[data-testid="stSlider"] [data-testid="stTrack"]{
  background: rgba(0,0,0,.12);
}
[data-testid="stSlider"] [data-testid="stTrack"] > div{
  background: var(--primary);
}

/* -------------------- INPUT NUMBER / TEXT -------------------- */
.stNumberInput input, .stTextInput input{
  background: rgba(255,255,255,0.6) !important;
  border-radius: var(--radius) !important;
  border: 1px solid rgba(0,0,0,.05) !important;
  box-shadow: inset 0 2px 4px rgba(0,0,0,.05) !important;
}

/* -------------------- BOUTON -------------------- */
.stButton>button{
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius) !important;
  padding: .65rem 1.4rem !important;
  font-weight: 600 !important;
  letter-spacing: .02em;
  transition: all .22s ease;
  box-shadow: 0 8px 18px rgba(90,123,255,.28);
}
.stButton>button:hover{
  background: var(--primary-light) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 24px rgba(90,123,255,.32);
}
.stButton>button:active{
  transform: translateY(0) scale(.98) !important;
}

/* Petit bouton (col6) */
[class*="stColumn"]:nth-child(7) .stButton>button{
  padding: .55rem .9rem !important;
  font-size: .85rem !important;
}

/* -------------------- TOOLTIPS -------------------- */
[data-baseweb="tooltip"]{
  backdrop-filter: blur(12px);
  background: rgba(0,0,0,.75);
  color: #fff;
  border-radius: 8px;
  font-size: .75rem;
  padding: .4rem .65rem;
}

/* -------------------- SIDEBAR -------------------- */
.sidebar .block-container{
  padding: 1rem 1rem 2rem 1rem !important;
}
[class*="stSidebar"]{
  background: rgba(255,255,255,0.7) !important;
  backdrop-filter: blur(18px);
  box-shadow: var(--shadow);
}

/* -------------------- CHARTS -------------------- */
.js-plotly-plot .plotly .main-svg{
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

/* -------------------- SCROLLBAR -------------------- */
::-webkit-scrollbar{
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-thumb{
  background: var(--primary-light);
  border-radius: 10px;
}
::-webkit-scrollbar-track{
  background: transparent;
}

/* -------------------- SLIDER -------------------- */
/* Cacher totalement les chiffres min/max + graduations */
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{
    display:none !important;
}

stSliderTickBarMin

/* -------------------- MASQUER MENU & FOOTER -------------------- */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* -------------------- GRADIENT TEXT -------------------- */
.gradient-premium {
  font-size: 2.5rem !important;             /* Titre XXL */
  font-weight: 800 !important;              /* Plus de présence */
  letter-spacing: -0.025em !important;      /* Ajustement espacement */

  /* Dégradé en trois couleurs */
  background: linear-gradient(
    360deg,
    #5A7BFF 10%,
    #5A7BFF 100%,
    #F0F4FF 150%
  ) !important;
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;

  /* contour/glow léger */
  text-shadow:
    0 0 2px rgba(255,255,255,0.8)

  display: inline-block;
}
</style>
"""


st.markdown(css, unsafe_allow_html=True)


def show(
    config_path: dict, 
    height: int=600
):

    # Chargement des config
    params_config = pipeline_config(config_path, type="stat")
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
    scale_choice = params_config["scale_choice"]
    show_relief = params_config["show_relief"]
    show_stations = params_config["show_stations"]

    # Préparation des paramètres pour pipeline_data
    params_load = (
        stat_choice_key,
        scale_choice_key,
        min_year_choice,
        max_year_choice,
        season_choice_key,
        missing_rate,
        quantile_choice,
        scale_choice
    )

    # Obtention des données
    result = pipeline_data(params_load, config, use_cache=True)

    # Chargement des affichages graphiques
    unit_label = get_stat_unit(stat_choice_key, scale_choice_key)
    params_map = (
        stat_choice_key,
        result,
        unit_label,
        height
    )
    layer, scatter_layer, tooltip, view_state, html_legend = pipeline_map(params_map)
    
    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        scatter_layer = None if not show_stations else scatter_layer
        deck = plot_map([layer, scatter_layer], view_state, tooltip, activate_relief=show_relief)
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

    with col2:
        st.markdown(html_legend, unsafe_allow_html=True)        

    with col3:
        params_scatter = (
            result,
            stat_choice_key, 
            scale_choice_key, 
            stat_choice,unit_label, 
            height
        )
        n_tot_mod, n_tot_obs, me, mae, rmse, r2, scatter = pipeline_scatter(params_scatter)

        st.markdown(
            """
            <div style='text-align: left; font-size: 0.8em; color: grey; margin-top: 0px;'>
                Données CP-RCM, 2.5 km, forçage ERA5, réanalyse ECMWF
            </div>
            """,
            unsafe_allow_html=True
        )
        st.plotly_chart(scatter, use_container_width=True)

    col0bis, col1bis, col2bis, col3bis, col4bis, col5bis, col6bis = st.columns(7)

    show_info_data(col0bis, "CP-AROME map", result["modelised_show"].shape[0], n_tot_mod)
    show_info_data(col1bis, "Stations", result["observed_show"].shape[0], n_tot_obs)
    show_info_data(col2bis, "CP-AROME plot", result["modelised"].shape[0], n_tot_mod)
    show_info_metric(col3bis, "ME", me)
    show_info_metric(col4bis, "MAE", mae)
    show_info_metric(col5bis, "RMSE", rmse)
    show_info_metric(col6bis, "r²", r2)

if __name__ == "__main__":
    config_path = "app/config/config.yaml"
    st.markdown("""
      <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
          font-family: var(--font);
          margin: 0;
        ">
          <span class="gradient-premium">
            Analyse interactive des précipitations en France — 1959 – 2022
          </span>
        </h1>
      </div>
    """, unsafe_allow_html=True)

    show(config_path)
