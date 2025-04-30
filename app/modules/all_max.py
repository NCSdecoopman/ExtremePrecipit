import streamlit as st
import polars as pl

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.utils.data_utils import add_metadata
from app.utils.stats_utils import generate_metrics
from app.utils.config_utils import load_config, menu_config

from app.utils.map_utils import plot_map
from app.utils.legends_utils import get_stat_unit

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter
def pipeline_config(config_path): 
    config = load_config(config_path)
    STATS, SEASON, SCALE = menu_config()

    return {
        "config": config,
        "stat_choice": "Maximum",
        "season_choice": "hydro",
        "stat_choice_key": STATS["Maximum"],
        "scale_choice_key": SCALE["Horaire"],
        "min_year_choice": config["years"]["min"],
        "max_year_choice": config["years"]["max"],
        "season_choice_key": SEASON["Année hydrologique"],
        "missing_rate": 0.15,
        "quantile_choice": 1
    }


def show(config_path):
    st.title("🔬 Tableau scientifique des performances")
    st.markdown(
        """
        Ce tableau présente les performances de la modélisation pour différentes saisons, échelles temporelles
        et périodes d'analyse. Les métriques incluent :
        - **ME** : Erreur moyenne (Mean Error)
        - **r²** : Coefficient de détermination
        """
    )

    saisons = {
        "Année hydrologique": "hydro",
        "Hiver": "djf",
        "Printemps": "mam",
        "Été": "jja",
        "Automne": "son"
    }
    echelles = {
        "Horaire": "mm_h",
        "Journalière": "mm_j"
    }
    periodes = {
        "mm_h": [(2000, 2015)],
        "mm_j": [(1960, 2015), (2000, 2015)]
    }

    params_config = pipeline_config(config_path)

    results = []
    results_grouped = []

    for nom_saison, saison_key in saisons.items():
        for nom_echelle, echelle_key in echelles.items():
            for min_year, max_year in periodes[echelle_key]:
                config = params_config["config"]
                stat_choice_key = "mean-max"
                quantile_choice = 1
                missing_rate = 0.15
                column_to_show = f"max_{echelle_key}"

                params_load = (
                    stat_choice_key,
                    echelle_key,
                    min_year,
                    max_year,
                    saison_key,
                    missing_rate,
                    quantile_choice
                )


                result = pipeline_data(params_load, config)
                df_modelised_load = add_metadata(result["modelised_load"], echelle_key, type="modelised")
                observed_cleaning = add_metadata(result["observed_cleaning"], echelle_key, type="observed")

                # Mapping NUM_POSTE_obs → NUM_POSTE_mod
                try:
                    csv_path = f"data/metadonnees/obs_vs_mod/obs_vs_mod_{"horaire" if echelle_key == "mm_h" else "quotidien"}.csv"
                    df_mapping = pl.read_csv(csv_path)
                except FileNotFoundError:
                    st.error(f"Fichier de correspondance non trouvé : {csv_path}")
                    return

                # Sélection des colonnes utiles
                df_obs = observed_cleaning.select(["NUM_POSTE", "year", column_to_show]).rename({
                    "NUM_POSTE": "NUM_POSTE_obs",
                    column_to_show: "pr_obs"
                })
                df_mod = df_modelised_load.select(["NUM_POSTE", "year", column_to_show]).rename({
                    "NUM_POSTE": "NUM_POSTE_mod",
                    column_to_show: "pr_mod"
                })

                # Jointure 1 : obs + mapping
                df = df_mapping.join(df_obs, on="NUM_POSTE_obs", how="left")

                # Jointure 2 : mod + mapping
                df = df.join(df_mod, on=["NUM_POSTE_mod", "year"], how="left")

                df = df.filter(df["pr_obs"].is_not_null() & df["pr_mod"].is_not_null())
      
                me, _, _, r2 = generate_metrics(df)
                results.append({
                    "Echelle": nom_echelle,
                    "Saison": nom_saison,
                    "Période": f"{min_year} - {max_year}",
                    "Stations": df["NUM_POSTE_obs"].n_unique(),
                    "r²": round(r2, 2),
                    "ME": round(me, 2)
                })


                params_config = pipeline_config(config_path)
                df_modelised_load = result["modelised_load"]
                df_observed_load = result["observed_load"]
                result_df_modelised_show = result["modelised_show"]
                result_df_modelised = result["modelised"]
                result_df_observed = result["observed"]
                column_to_show = result["column"]
                stat_choice = params_config["stat_choice"]
                season_choice = params_config["season_choice"]
                stat_choice_key = params_config["stat_choice_key"]
                scale_choice_key = params_config["scale_choice_key"]
                min_year_choice = params_config["min_year_choice"]
                max_year_choice = params_config["max_year_choice"]
                season_choice_key = params_config["season_choice_key"]
                missing_rate = params_config["missing_rate"]
                quantile_choice = params_config["quantile_choice"]




                params_scatter = (
                    df_modelised_load, 
                    df_observed_load, 
                    column_to_show, 
                    result_df_modelised, 
                    result_df_modelised_show, 
                    result_df_observed, 
                    stat_choice_key, 
                    scale_choice_key, 
                    stat_choice,"", 
                    0
                )
                me_grouped, r2_grouped = pipeline_scatter(params_scatter)

                # Calcul des métriques à partir des moyennes par station
                results_grouped.append({
                    "Echelle": nom_echelle,
                    "Saison": nom_saison,
                    "Période": f"{min_year} - {max_year}",
                    "Stations": "",
                    "r²": round(r2_grouped, 2),
                    "ME": round(me_grouped, 2)
                })



    if results and results_grouped:
        # Exemple avec ta variable `results`
        df = pl.DataFrame(results)
        df_grouped = pl.DataFrame(results_grouped)

        # Ordre voulu pour les saisons
        ordered_seasons = ["Année hydrologique", "Hiver", "Printemps", "Été", "Automne"]

        # Création d'une colonne d'ordre temporaire
        df = df.with_columns([
            pl.when(pl.col("Saison") == ordered_seasons[0]).then(0)
            .when(pl.col("Saison") == ordered_seasons[1]).then(1)
            .when(pl.col("Saison") == ordered_seasons[2]).then(2)
            .when(pl.col("Saison") == ordered_seasons[3]).then(3)
            .when(pl.col("Saison") == ordered_seasons[4]).then(4)
            .otherwise(99)
            .alias("saison_order")
        ])

        df_grouped = df_grouped.with_columns([
            pl.when(pl.col("Saison") == ordered_seasons[0]).then(0)
            .when(pl.col("Saison") == ordered_seasons[1]).then(1)
            .when(pl.col("Saison") == ordered_seasons[2]).then(2)
            .when(pl.col("Saison") == ordered_seasons[3]).then(3)
            .when(pl.col("Saison") == ordered_seasons[4]).then(4)
            .otherwise(99)
            .alias("saison_order")
        ])

        # Tri en utilisant la nouvelle colonne
        df = df.sort(["Echelle", "Période", "saison_order"])
        df_grouped = df_grouped.sort(["Echelle", "Période", "saison_order"])


        # Regrouper les données sous forme de colonnes multiples
        rows = df.select(["Echelle", "Période"]).unique()
        rows_grouped = df_grouped.select(["Echelle", "Période"]).unique()

        for saison in ordered_seasons:
            df_s = df.filter(pl.col("Saison") == saison).select([
                "Echelle", "Période", 
                pl.col("Stations").alias(f"{saison}_Stations"),
                pl.col("r²").alias(f"{saison}_r²"),
                pl.col("ME").alias(f"{saison}_ME")
            ])
            rows = rows.join(df_s, on=["Echelle", "Période"], how="left")

            df_s_grouped = df_grouped.filter(pl.col("Saison") == saison).select([
                "Echelle", "Période", 
                pl.col("Stations").alias(f"{saison}_Stations"),
                pl.col("r²").alias(f"{saison}_r²"),
                pl.col("ME").alias(f"{saison}_ME")
            ])
            rows_grouped = rows_grouped.join(df_s_grouped, on=["Echelle", "Période"], how="left")

        # Affichage dans Streamlit
        st.dataframe(rows)
        st.dataframe(rows_grouped)

    else:
        st.warning("Aucune donnée appariée sur la période.")
