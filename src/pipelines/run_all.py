# run_all.py
import subprocess
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__).info

############################################################
# ------------------------- STATS -------------------------
############################################################
# # Pipeline des données AROME
# log(f"Lancement du traitement .nc to .zarr (echelle horaire obligatoire)")
# subprocess.run(
#     ["python", "-m", "src.pipelines.pipeline_nc_to_zarr"],
#     check=True
# )

# log(f"Lancement du traitement .zarr MODELISE to stats (echelle horaire obligatoire)")
# subprocess.run(
#     ["python",
#         "-m",
#         f"src.pipelines.pipeline_zarr_to_stats",
#         "--config", "config/modelised_settings.yaml",
#         "--echelle", "horaire"],
#     check=True
# )


## Pipeline des données STATIONS
# PIPELINE_TO_RUN_OBS = ["obs_to_zarr", "zarr_to_stats"]
# ECHELLES = ["horaire", "quotidien"]

# for pipeline in PIPELINE_TO_RUN_OBS:
#     for echelle in ECHELLES:
#         log(f"Lancement du traitement {pipeline} pour l’échelle : {echelle}")
#         subprocess.run(
#             ["python",
#              "-m",
#              f"src.pipelines.pipeline_{pipeline}",
#              "--config", "config/observed_settings.yaml",
#              "--echelle", f"{echelle}"],
#             check=True
#         )


############################################################
# ---------------------- SCATTER PLOT ----------------------
############################################################
# # Pipeline obs vs mod
# for echelle in ["horaire", "quotidien"]:
#     log(f"Lancement du traitement obs vs mod (echelle {echelle})")
#     subprocess.run(
#         ["python", 
#          "-m", 
#          "src.pipelines.pipeline_obs_vs_mod",
#          "--echelle", f"{echelle}"],
#         check=True
#     )


############################################################
# -------------------------- GEV --------------------------
############################################################
# Pipeline GEV
log(f"Lancement du traitement stats to gev")
for setting in ["config/observed_settings.yaml", "config/modelised_settings.yaml"]:

    for echelle in ["quotidien"]:

        for season in ["hydro", "djf", "mam", "jja", "son"]:

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
                    "src.pipelines.pipeline_best_gev",
                    "--config", setting,
                    "--echelle", echelle,
                    "--season", season
                ],
                check=True
            )



# A INTEGRER DANS L'AUTRE

log(f"Lancement du traitement stats to gev")
for setting in ["config/observed_settings.yaml", "config/modelised_settings.yaml"]:

    for echelle in ["horaire"]:

        for season in ["hydro", "djf", "mam", "jja", "son"]:

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
                    "src.pipelines.pipeline_best_gev",
                    "--config", setting,
                    "--echelle", echelle,
                    "--season", season
                ],
                check=True
            )
