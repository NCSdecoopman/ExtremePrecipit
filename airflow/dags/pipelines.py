from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import timedelta

default_args = {
    "owner": "nicolas",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="zarr_stats_gev_pipeline",
    default_args=default_args,
    description="Pipeline Zarr → Stats → GEV",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["meteo"],
) as dag:

    # ──────────────── ZARR ────────────────
    zarr_arome = BashOperator(
        task_id="zarr_arome",
        bash_command="python -m src.pipelines.pipeline_nc_to_zarr"
    )

    zarr_obs_h = BashOperator(
        task_id="zarr_obs_horaire",
        bash_command="python -m src.pipelines.pipeline_obs_to_zarr --echelle horaire"
    )
    zarr_obs_q = BashOperator(
        task_id="zarr_obs_quotidien",
        bash_command="python -m src.pipelines.pipeline_obs_to_zarr --echelle quotidien"
    )

    # ──────────────── MÉTADONNÉES ────────────────
    with TaskGroup("metadata") as metadata:
        for echelle in ["horaire", "quotidien"]:
            BashOperator(
                task_id=f"metadata_{echelle}",
                bash_command=f"python -m src.pipelines.pipeline_obs_vs_mod --echelle {echelle}"
            )

    # ──────────────── AGRÉGATION ────────────────
    with TaskGroup("aggregation") as aggregation:
        for n in [3, 5]:
            BashOperator(
                task_id=f"aggregate_n{n}",
                bash_command=f"python -m src.pipelines.pipeline_aggregate_to_zarr --n_aggregate {n}"
            )

    # ──────────────── STATS ────────────────
    with TaskGroup("stats") as stats:
        for config in ["config/modelised_settings.yaml", "config/observed_settings.yaml"]:
            ECHELLES = ["horaire_aggregate_n3"]
            if config == "config/observed_settings.yaml":
                ECHELLES.append("quotidien")

            for echelle in ECHELLES:
                BashOperator(
                    task_id=f"stats_{echelle}_{config.split('/')[-1].replace('.yaml', '')}",
                    bash_command=f"python -m src.pipelines.pipeline_zarr_to_stats --config {config} --echelle {echelle}"
                )

    # ──────────────── GEV ────────────────
    with TaskGroup("gev") as gev:
        for setting in ["config/modelised_settings.yaml"]:
            for echelle in ["quotidien", "horaire"]:
                for season in ["hydro", "djf", "mam", "jja", "son"]:
                    for model in [
                        "s_gev", "ns_gev_m1", "ns_gev_m2", "ns_gev_m3",
                        "ns_gev_m1_break_year", "ns_gev_m2_break_year", "ns_gev_m3_break_year",
                    ]:
                        BashOperator(
                            task_id=f"gev_{echelle}_{season}_{model}",
                            bash_command=(
                                f"python -m src.pipelines.pipeline_stats_to_gev "
                                f"--config {setting} --echelle {echelle} --season {season} --model {model}"
                            )
                        )

                    BashOperator(
                        task_id=f"best_model_{echelle}_{season}",
                        bash_command=(
                            f"python -m src.pipelines.pipeline_best_model "
                            f"--config {setting} --echelle {echelle} --season {season}"
                        )
                    )

    # ──────────────── DÉPENDANCES ────────────────
    [zarr_obs_h, zarr_obs_q, zarr_arome] >> metadata
    metadata >> aggregation >> stats
    metadata >> stats
    stats >> gev
