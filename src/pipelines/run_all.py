# run_all.py
import subprocess
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__).info

PIPELINE_TO_RUN_MOD = ["nc_to_zarr", "zarr_to_stats"]
PIPELINE_TO_RUN_OBS = ["obs_to_zarr", "zarr_to_stats"]
ECHELLES = ["horaire"]

# for pipeline in PIPELINE_TO_RUN_MOD:
#     for echelle in ["horaire"]:
#         log(f"Lancement du traitement {pipeline} pour l’échelle : {echelle}")
#         subprocess.run(
#             ["python",
#              "-m",
#              f"src.pipelines.pipeline_{pipeline}",
#              "--config", "config/modelised_settings.yaml",
#              "--echelle", echelle],
#             check=True
#         )

for pipeline in PIPELINE_TO_RUN_OBS:
    for echelle in ECHELLES:
        log(f"Lancement du traitement {pipeline} pour l’échelle : {echelle}")
        subprocess.run(
            ["python",
             "-m",
             f"src.pipelines.pipeline_{pipeline}",
             "--config", "config/observed_settings.yaml",
             "--echelle", echelle],
            check=True
        )