import streamlit as st
import polars as pl

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.utils.data_utils import add_metadata
from app.utils.stats_utils import generate_metrics
from app.utils.config_utils import load_config, menu_config_statisticals

from app.utils.map_utils import plot_map
from app.utils.legends_utils import get_stat_unit

from app.pipelines.import_data import pipeline_data
from app.pipelines.import_config import pipeline_config
from app.pipelines.import_map import pipeline_map
from app.pipelines.import_scatter import pipeline_scatter

import plotly.express as px
import pandas as pd

def pipeline_config(config_path: dict): 
    config = load_config(config_path)
    STATS, SEASON, SCALE = menu_config_statisticals()

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


def show(config_path, min_year: int, max_year: int, middle_year: int, missing_rate: int):
    st.title("Tableau scientifique des performances")
    st.markdown(
        """
        Ce tableau présente les performances de la modélisation pour différentes saisons, échelles temporelles
        et périodes d'analyse. Les métriques incluent :
        - **ME** : Erreur moyenne (Mean Error)
        - **r²** : Coefficient de détermination
        """
    )

    saisons = {
        "Hydro": "hydro",
        "DJF": "djf",
        "MAM": "mam",
        "JJA": "jja",
        "SON": "son"
    }
    echelles = {
        "w3": "mm_h",
        "w6": "mm_h",
        "w9": "mm_h",
        "w12": "mm_h",
        "w24": "mm_h",
        "Horaire": "mm_h",
        # "horaire_aggregate_n3": "mm_h",
        # "horaire_aggregate_n5": "mm_h",
        "Journalière": "mm_j"
    }
    echelle_labels = {
        "w3": "3h glissante",
        "w6": "6h glissante",
        "w9": "9h glissante",
        "w12": "12h glissante",
        "w24": "24h glissante",
        "Horaire": "1h",
        # "horaire_aggregate_n3": "1h (agreg 3x3)",
        # "horaire_aggregate_n5": "1h (agreg 5x5)",
        "Journalière": "Journalière"
    }

    params_config = pipeline_config(config_path)

    periodes = {
        "mm_h": [(middle_year, max_year)],
        "mm_j": [(min_year, max_year)] #, (middle_year, max_year)
    }

    results = []
    results_grouped = []

    renvoi_final = []

    for nom_saison, saison_key in saisons.items():
        for nom_echelle, echelle_key in echelles.items():

            for min_year, max_year in periodes[echelle_key]:

                if nom_echelle == "Journalière" and len(periodes["mm_j"]) > 1:
                    echelle_affiche = f"{echelle_labels[nom_echelle]} ({min_year}-{max_year})"
                else:
                    echelle_affiche = echelle_labels[nom_echelle]

                config = params_config["config"]
                stat_choice_key = "mean-max"
                quantile_choice = 1
                column_to_show = f"max_{echelle_key}"

                if nom_echelle == "Journalière":
                    scale_choice = "quotidien"
                else:
                    scale_choice = nom_echelle.lower()

                params_load = (
                    stat_choice_key,
                    echelle_key,
                    min_year if saison_key not in ["hydro", "djf"] else min_year+1,
                    max_year,
                    saison_key,
                    missing_rate,
                    quantile_choice,
                    scale_choice
                )


                result = pipeline_data(params_load, config)
                df_modelised_load = add_metadata(result["modelised_load"], echelle_key, type="modelised")
                observed_cleaning = add_metadata(result["observed_cleaning"], echelle_key, type="observed")

                # Sélection des colonnes utiles
                df_obs = observed_cleaning.select(["NUM_POSTE", "year", column_to_show]).rename({
                    "NUM_POSTE": "NUM_POSTE_obs",
                    column_to_show: "Station"
                })
                df_mod = df_modelised_load.select(["NUM_POSTE", "year", column_to_show]).rename({
                    "NUM_POSTE": "NUM_POSTE_mod",
                    column_to_show: "AROME"
                })


                if "_aggregate_" in scale_choice:
                    # Hypothèse : les codes station NUM_POSTE sont identiques dans l’observation
                    # (observed_cleaning) et dans la modélisation (df_modelised_load). Il n’y a
                    # donc pas besoin du fichier de correspondance obs_vs_mod_*.csv.

                    # 2. Jointure directe sur NUM_POSTE + year
                    df = df_obs.join(
                        df_mod,
                        left_on=["NUM_POSTE_obs", "year"],
                        right_on=["NUM_POSTE_mod", "year"],
                        how="inner"
                    )

                    # 3. Nettoyage : on écarte les lignes avec valeurs manquantes
                    df = df.filter(df["Station"].is_not_null() & df["AROME"].is_not_null())


                else:
                    # Mapping NUM_POSTE_obs → NUM_POSTE_mod
                    try:
                        csv_path = f"data/metadonnees/obs_vs_mod/obs_vs_mod_{"horaire" if echelle_key == "mm_h" else "quotidien"}.csv"
                        df_mapping = pl.read_csv(csv_path)
                    except FileNotFoundError:
                        st.error(f"Fichier de correspondance non trouvé : {csv_path}")
                        return None

                    # Jointure 1 : obs + mapping
                    df = df_mapping.join(df_obs, on="NUM_POSTE_obs", how="left")

                    # Jointure 2 : mod + mapping
                    df = df.join(df_mod, on=["NUM_POSTE_mod", "year"], how="left")

                    df = df.filter(df["Station"].is_not_null() & df["AROME"].is_not_null())

                    if echelle_key == "mm_j" and saison_key == "hydro":
                        renvoi_final = df
                        
                me, _, _, r2 = generate_metrics(df)
                results.append({
                    "Echelle": echelle_affiche,
                    "Saison": nom_saison,
                    "Période": f"{min_year} - {max_year}",
                    "Stations": df["NUM_POSTE_obs"].n_unique(),
                    "r²": round(r2, 2),
                    "ME": round(me, 2)
                })


                if "_aggregate_" in scale_choice:
                    # Jointure directe sur NUM_POSTE
                    obs = result["observed"].rename({"max_mean_mm_h": "Station"})
                    mod = result["modelised"].rename({"max_mean_mm_h": "AROME"})

                    df = obs.join(mod, on="NUM_POSTE", how="inner", suffix="_mod")  # suffix si d’autres doublons
                    df = df.rename({"NUM_POSTE": "NUM_POSTE_obs"})
                    me_grouped, _, _, r2_grouped = generate_metrics(df)
 
                else:
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
                        result,
                        stat_choice_key, 
                        scale_choice_key, 
                        stat_choice,
                        "",# unit 
                        500 # height
                    )
                    _, _, me_grouped, _, _, r2_grouped, _ = pipeline_scatter(params_scatter)
                
                # Calcul des métriques à partir des moyennes par station
                results_grouped.append({
                    "Echelle": echelle_affiche,
                    "Saison": nom_saison,
                    "Période": f"{min_year} - {max_year}",
                    "Stations": df["NUM_POSTE_obs"].n_unique(),
                    "r²": round(r2_grouped, 2),
                    "ME": round(me_grouped, 2)
                })



    if results and results_grouped:
        # Exemple avec ta variable `results`
        df = pl.DataFrame(results)
        df_grouped = pl.DataFrame(results_grouped)

        # Ordre voulu pour les saisons
        ordered_seasons = ["Hydro", "DJF", "MAM", "JJA", "SON"]

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




        # Remplacer ceci par le chemin vers tes fichiers ou l’objet déjà chargé
        df_annuel = rows_grouped
        df_moyenne = rows
        # Liste des colonnes communes (tu peux l'étendre selon besoin)
        common_columns = df_annuel.columns

        # On cast toutes les colonnes de df_moyenne pour matcher les types de df_annuel
        schema_annuel = {col: df_annuel.schema[col] for col in common_columns}

        # Fonction de cast uniforme
        def cast_to_schema(df, schema):
            return df.with_columns([
                pl.col(col).cast(dtype) for col, dtype in schema.items()
            ])

        df_annuel = cast_to_schema(df_annuel, schema_annuel)
        df_moyenne = cast_to_schema(df_moyenne, schema_annuel)

        # Ajout des colonnes Type
        df_annuel = df_annuel.with_columns(pl.lit("Moyenne des maxima").alias("Type"))
        df_moyenne = df_moyenne.with_columns(pl.lit("Maxima annuels").alias("Type"))

        # Concaténation
        df_all = pl.concat([df_annuel, df_moyenne])

        # Passage à pandas pour Plotly
        df_all = df_all.to_pandas()

        def melt_metrics(df, metric):
            """Transforme le DataFrame pour Plotly en incluant 'Stations'."""
            melted = pd.DataFrame()
            seasons = ["Hydro", "DJF", "MAM", "JJA", "SON"]
            for saison in seasons:
                col_metric = f"{saison}_{metric}"
                col_stations = f"{saison}_Stations"

                if col_metric in df.columns:
                    temp = df[["Echelle", "Période", "Type"]].copy()
                    temp["Saison"] = saison
                    temp[metric] = df[col_metric]
                    if col_stations in df.columns:
                        temp["Stations"] = df[col_stations]
                    else:
                        temp["Stations"] = None
                    melted = pd.concat([melted, temp], ignore_index=True)
            return melted


        df_r2 = melt_metrics(df_all, "r²")
        df_me = melt_metrics(df_all, "ME")

        # Affichage dans Streamlit
        st.subheader("Comparaison des r² par saison et type")
        fig_r2 = px.bar(
            df_r2,
            x="Saison",
            y="r²",
            color="Type",
            barmode="group",
            facet_col="Echelle",
            title="r² par saison",
            height=500
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        st.subheader("Comparaison des erreurs moyennes (ME) par saison et type")
        fig_me = px.bar(
            df_me,
            x="Saison",
            y="ME",
            color="Type",
            barmode="group",
            facet_col="Echelle",
            title="Erreur moyenne (ME) par saison",
            height=500
        )
        st.plotly_chart(fig_me, use_container_width=True)

        df_r2 = df_r2.merge(df_me[["Echelle", "Période", "Type", "Saison", "ME"]], 
                            on=["Echelle", "Période", "Type", "Saison"], 
                            how="left")

        return renvoi_final, df_r2

    else:
        st.warning("Aucune donnée appariée sur la période.")

