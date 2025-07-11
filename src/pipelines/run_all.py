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
    "hydro",
    "son",
    "djf",
    "mam",
    "jja"
]


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

# for config in ["config/observed_settings.yaml", "config/modelised_settings.yaml"]:
    
#     # "horaire", "w3", "w6", "w9", "w12", "w24"]
#     SEASONS = SEASON_MONTHS
    
#     if config == "config/observed_settings.yaml": # Pas de temps horaire uniquement pour AROME
#         ECHELLES  = ["horaire", "quotidien"]
#     else:
#         ECHELLES  = ["horaire"]

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

# Pipeline GEV
log(f"Lancement du traitement stats to gev")
for setting in ["config/observed_settings.yaml"]:#, "config/modelised_settings.yaml"

    for echelle in ["quotidien"]:

        for season in ["sep", "oct", "nov", "dec"]: 

            for model in [
                "s_gev", # Stationnaire
                "ns_gev_m1", "ns_gev_m2", "ns_gev_m3", # Non stationnaire
                "ns_gev_m1_break_year", "ns_gev_m2_break_year", "ns_gev_m3_break_year" # Non stationnaire avec point de rupture
                ]:
            
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "src.pipelines.pipeline_stats_to_gev",
                        "--config", setting,
                        "--echelle", echelle,
                        "--season", season,
                        "--model", model
                    ],
                    check=True
                )

            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.pipelines.pipeline_best_model",
                    "--config", setting,
                    "--echelle", echelle,
                    "--season", season
                ],
                check=True
            )


            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.pipelines.pipeline_best_to_niveau_retour",
                    "--config", setting,
                    "--echelle", echelle,
                    "--season", season
                ],
                check=True
            )