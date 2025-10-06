# run_all.py
import subprocess
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__).info

SEASON_MONTHS = [
    "jan",
    "fev",
    "mar",
    "avr",
    "mai",
    "jui",
    "juill",
    "aou",
    "sep",
    "oct",
    "nov",
    "dec"
]

SEASON_SEAS = [
    "djf", "jfm",
    "mam", "amj",
    "jja", "jas",
    "son", "ond"
]

SEASONS = SEASON_SEAS + SEASON_MONTHS


############################################################
# ------------------------- ZARR -------------------------
############################################################

# # Pipeline des données AROME
# log(f"Lancement du traitement .nc to .zarr (echelle horaire obligatoire)")
# subprocess.run(
#     ["python", 
#       "-m", 
#       "src.pipelines.pipeline_nc_to_zarr"],
#     check=True
# )

# for echelle in ["horaire, "quotidien""]: 
#     log(f"Lancement du traitement obs (.csv) to .zarr pour l’échelle : {echelle}")
#     subprocess.run(
#         ["python",
#             "-m",
#             f"src.pipelines.pipeline_obs_to_zarr",
#             "--echelle", f"{echelle}"],
#         check=True
#     )


############################################################
# ------------------------- METADONNES ---------------------
############################################################

# log(f"Lancement des metadonnées obs vs mod")
# for echelle in ["horaire", "quotidien"]:
#     subprocess.run(
#         ["python", 
#         "-m", 
#         "src.pipelines.pipeline_obs_vs_mod",
#         "--echelle", echelle],
#         check=True
#     )


############################################################
# ------------------------- AGGREGATION SPATIALE -----------
############################################################

# log("Lancement des aggrégations spatiales")
# for n_aggregate in [3, 5]: # nombre impair
#     subprocess.run(
#         ["python", 
#          "-m", 
#          "src.pipelines.pipeline_aggregate_to_zarr",
#          "--n_aggregate", str(n_aggregate)],
#         check=True
#     )

############################################################
# ------------------------- STATS -------------------------
############################################################

# for config in ["config/modelised_settings.yaml", "config/observed_settings.yaml"]:
       
#     if config == "config/observed_settings.yaml": # Pas de temps horaire uniquement pour AROME
#         ECHELLES  = ["horaire", "quotidien"] # "w3", "w6", "w9", "w12", "w24", 
#     else:
#         ECHELLES  = ["horaire"] # "w3", "w6", "w9", "w12", "w24"

#     for echelle in ECHELLES:
#         for season in SEASONS:
#             log(f"Lancement du traitement .zarr to stats {config} - {echelle} -{season}")
#             subprocess.run(
#                 ["python",
#                 "-m",
#                 "src.pipelines.pipeline_zarr_to_stats",
#                 "--config", config,
#                 "--echelle", echelle,
#                 "--season", season],
#             check=True
#         )



# ############################################################
# # -------------------------- GEV --------------------------
# ############################################################

# # Pipeline GEV
# log(f"Lancement du traitement gev")

# for setting in ["config/observed_settings.yaml", "config/modelised_settings.yaml"]: #, ""

#     for echelle in ["horaire"]: #"quotidien", 

#         if echelle == "quotidien":            
#             DIFFERENTE_PERIODE = [False, True]
#         else:            
#             DIFFERENTE_PERIODE = [False] # True
            
        
#         for diffente_periode in DIFFERENTE_PERIODE:
        
#             if echelle == "quotidien" and not diffente_periode:
#                 MODELS = [
#                     "s_gev", # Stationnaire
#                     "ns_gev_m1", "ns_gev_m2", "ns_gev_m3", # Non stationnaire
#                     "ns_gev_m1_break_year", "ns_gev_m2_break_year", "ns_gev_m3_break_year" # Non stationnaire avec point de rupture
#                     ]
#             elif echelle == "horaire" and diffente_periode:
#                 MODELS = [
#                     "s_gev", # Stationnaire
#                     "ns_gev_m1", "ns_gev_m2", "ns_gev_m3", # Non stationnaire
#                     "ns_gev_m1_break_year", "ns_gev_m2_break_year", "ns_gev_m3_break_year" # Non stationnaire avec point de rupture
#                     ]
#             else:
#                 MODELS = [
#                     "s_gev", # Stationnaire
#                     "ns_gev_m1", "ns_gev_m2", "ns_gev_m3", # Non stationnaire
#                     ]

#             for season in SEASONS:
#                 for model in MODELS:
                
#                     subprocess.run(
#                         [
#                             "python",
#                             "-m",
#                             "src.pipelines.pipeline_stats_to_gev",
#                             "--config", setting,
#                             "--echelle", echelle,
#                             "--season", season,
#                             "--model", model,
#                             "--reduce_activate", str(diffente_periode)
#                         ],
#                         check=True
#                     )

#                 subprocess.run(
#                     [
#                         "python",
#                         "-m",
#                         "src.pipelines.pipeline_best_model",
#                         "--config", setting,
#                         "--echelle", echelle,
#                         "--season", season,
#                         "--reduce_activate", str(diffente_periode)
#                     ],
#                     check=True
#                 )


#                 subprocess.run(
#                     [
#                         "python",
#                         "-m",
#                         "src.pipelines.pipeline_best_to_niveau_retour",
#                         "--config", setting,
#                         "--echelle", echelle,
#                         "--season", season,
#                         "--reduce_activate", str(diffente_periode)
#                     ],
#                     check=True
#                 )



############################################################
# -------------------------- MAPPING -----------------------
############################################################

# # Pipeline maps
log(f"Lancement des générations de maps")

for data_type in ["stats", "gev"]: # "dispo", "stats",  "gev"
                
    if data_type == "dispo":
        COL_CALCULATE = ["n_years"]
    elif data_type == "stats":
        COL_CALCULATE = [ "numday", "mean", "mean-max"] #
    elif data_type == "gev":
        COL_CALCULATE = ["z_T_p"] # "significant", , "model"

    for col_calculate in COL_CALCULATE:

        if col_calculate in ["mean", "mean-max"]:
            sat = 99
        elif col_calculate in ["numday"]:
            sat = 99.9
        elif col_calculate in ["z_T_p"]:
            sat = 99
        else:
            sat = 100
                
        for echelle in ["quotidien", "horaire"]: # 

            if col_calculate in ["z_T_p"]:
                if echelle=="horaire":
                    sat = 90

            if echelle == "quotidien":            
                DIFFERENTE_PERIODE = [False] # 
            else:            
                DIFFERENTE_PERIODE = [False] # 

            for diffente_periode in DIFFERENTE_PERIODE:

                if data_type == "dispo":
                    SEASON_GENERATE = [["hydro"]]
                elif data_type == "stats":
                    if col_calculate == "mean-max":
                        SEASON_GENERATE = [["hydro", *SEASON_SEAS]]
                    else:
                        SEASON_GENERATE = [["hydro"], SEASON_SEAS] # hydro doit être calculer séparement
                else:
                    SEASON_GENERATE = [["hydro", *SEASON_SEAS], SEASON_MONTHS]

                for s in SEASON_GENERATE:
                
                    subprocess.run(
                        [
                            "python",
                            "-m",
                            "src.pipelines.pipeline_generate_outputs",
                            "--data_type", data_type,
                            "--col_calculate", col_calculate,
                            "--echelle", echelle,
                            "--season", *s,
                            "--reduce_activate", str(diffente_periode),
                            "--sat", str(sat)
                        ],
                        check=True
                    )



# subprocess.run(
#     ["quarto", "render", "article/article_2.qmd"],
#     check=True
# )