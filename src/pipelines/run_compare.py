import subprocess
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__).info


for echelle in ["quotidien", "horaire"]:
    for season in ["hydro", "son", "djf", "mam", "jja"]:
        subprocess.run(
            ["python",
                "-m",
                f"src.pipelines.compare_date",
                "--echelle", echelle,
                "--season", season],
            check=True
        )