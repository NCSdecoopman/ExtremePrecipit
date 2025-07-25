import argparse
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from app.utils.data_utils import add_metadata
from app.utils.data_utils import match_and_compare

import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

def import_data_gev(
    season,
    echelle,
    reduce_activate,
    col_calculate,
    config_mod,
    config_obs,
    scale
):
    gev_dir_mod = config_mod["gev"]["path"]["outputdir"]
    gev_dir_obs = config_obs["gev"]["path"]["outputdir"]
    suffix_save = "_reduce" if reduce_activate and echelle == "quotidien" else ""

    for gev_dir in [gev_dir_mod, gev_dir_obs]:
        path_dir = Path(gev_dir) / f"{echelle}{suffix_save}" / season / "niveau_retour.parquet"
        
        if "modelised" in str(gev_dir):
            modelised = pl.read_parquet(path_dir)
            modelised = add_metadata(modelised, scale, type='modelised')
        elif "observed" in str(gev_dir):
            observed = pl.read_parquet(path_dir)
            observed = add_metadata(observed, scale, type='observed')          
     
    return {
        "modelised": modelised,
        "observed": observed,
        "column": col_calculate,
        "season": season
    }



def generate_violin(
    datasets,
    dir_path: Path,
    col_calculate: str,
    echelle: str,
    show_signif: bool
):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    violin_rows = []

    for e in echelle:
        for d in datasets[e]:
            mod = d.get("modelised")    # pl.DataFrame
            obs = d.get("observed")     # pl.DataFrame
            col = d.get("column")
            season = d.get("season")

            if show_signif:
                mod = mod.filter(pl.col("significant") == True)
                obs = obs.filter(pl.col("significant") == True)

            echelle_compare = "horaire" if e == "horaire" else "quotidien"
            df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{echelle_compare}.csv")
            obs_vs_mod = match_and_compare(obs, mod, col, df_obs_vs_mod)

            if obs_vs_mod.is_empty():
                logger.warning(f"Aucune donnée après filtrage pour {echelle} {season} - {col_calculate} - signif {show_signif}")
                continue

            # ----- COLLECTE POUR VIOLIN PLOT -----
            arome_vals = obs_vs_mod["AROME"].to_numpy()
            station_vals = obs_vs_mod["Station"].to_numpy()

            for v in arome_vals:
                violin_rows.append({"season": season, "echelle": e, "source": "AROME", "value": v})
            for v in station_vals:
                violin_rows.append({"season": season, "echelle": e, "source": "Stations", "value": v})

    suffix = "_signif" if show_signif else ""

    # violin + boxplot
    if violin_rows:
        import numpy as np
        df_violin = pd.DataFrame(violin_rows)
        seasons = df_violin["season"].unique()
        echelles = df_violin["echelle"].unique()
        counts = df_violin[df_violin.source == "AROME"].groupby(["season", "echelle"]).size()
        ymin, ymax = df_violin["value"].min(), df_violin["value"].max()
        offset = (ymax - ymin) * 0.20

        # Création de la colonne combinée pour l'axe x
        df_violin['saison_echelle'] = df_violin['season'] + ' - ' + df_violin['echelle']

        # Toujours 6 saisons/mois par ligne
        max_saisons_par_ligne = 6
        n_saisons = len(seasons)
        # Gestion insertion bloc vide entre jja et jan si besoin
        saisons_liste = list(seasons)
        if 'jja' in saisons_liste and 'jan' in saisons_liste:
            idx_jja = saisons_liste.index('jja')
            idx_jan = saisons_liste.index('jan')
            if idx_jan == idx_jja + 1:
                # On insère un bloc vide entre jja et jan
                saisons_liste.insert(idx_jja + 1, '__VIDE__')
        n_saisons_mod = len(saisons_liste)
        n_rows = (n_saisons_mod + max_saisons_par_ligne - 1) // max_saisons_par_ligne

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=1,
            figsize=(max(12, max_saisons_par_ligne*2.2), 3.5*n_rows),
            squeeze=False
            # sharey=True supprimé
        )

        for row in range(n_rows):
            start = row * max_saisons_par_ligne
            end = min((row+1)*max_saisons_par_ligne, n_saisons_mod)
            saisons_ligne = saisons_liste[start:end]
            order = []
            for s in saisons_ligne:
                if s == '__VIDE__':
                    for e in echelles:
                        order.append(f"__VIDE__ - {e}")
                else:
                    for e in echelles:
                        order.append(f"{s} - {e}")
            ax = axes[row, 0]
            data_row = df_violin[df_violin['season'].isin([s for s in saisons_ligne if s != '__VIDE__'])]

            # Ajouter explicitement des groupes vides pour les blocs __VIDE__
            for s in saisons_ligne:
                if s == '__VIDE__':
                    for e in echelles:
                        # Ajoute une ligne vide pour ce groupe
                        data_row = pd.concat([
                            data_row,
                            pd.DataFrame([{
                                'season': s,
                                'echelle': e,
                                'source': df_violin['source'].unique()[0],  # n'importe laquelle, ne sera pas affichée
                                'value': np.nan,
                                'saison_echelle': f"{s} - {e}"
                            }])
                        ], ignore_index=True)

            # Adapter l'axe y à chaque ligne (ymin fixé à -120, ymax à 400)
            ymin_ligne = -200
            ymax_ligne = 400

            sns.boxplot(
                data=data_row,
                x='saison_echelle',
                y='value',
                hue='source',
                order=order,
                dodge=True,
                ax=ax
            )
            ax.set_ylim(ymin_ligne, ymax_ligne)

            # Traits pleins entre chaque saison
            for i in range(1, len(saisons_ligne)):
                ax.axvline(i*len(echelles) - 0.5, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
            # Traits pointillés entre chaque échelle à l'intérieur d'une saison
            for i in range(len(saisons_ligne)):
                for j in range(1, len(echelles)):
                    ax.axvline(i*len(echelles) + j - 0.5, color='grey', linestyle=':', linewidth=1, alpha=0.7, zorder=0)
            # Grille horizontale pointillée sur les y
            ax.yaxis.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
            # Ligne horizontale sur 0
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=1, zorder=5)
            # Légende (seulement sur la première ligne)
            if row == 0:
                handles, labels = ax.get_legend_handles_labels()
                leg = ax.legend(handles, labels, title='', loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0, frameon=True)
                leg.set_zorder(20)
                leg.set_alpha(0.9)
                leg.get_frame().set_facecolor('white')
            else:
                ax.get_legend().remove()
            # Annotations n (un seul n= par sous-groupe saison-échelle, total des deux sources)
            for i, s in enumerate(saisons_ligne):
                if s == '__VIDE__':
                    continue
                for j, e in enumerate(echelles):
                    x_pos = i*len(echelles) + j
                    subset = data_row[(data_row['season'] == s) & (data_row['echelle'] == e)]
                    n = int(len(subset)/2)  # données AROME et obs
                    n_sup_400 = (subset['value'] > 400).sum()
                    n_inf_m200 = (subset['value'] < -200).sum()
                    if n > 0:
                        y_n = ymax_ligne - 0.02 * (ymax_ligne - ymin_ligne)
                        txt = f"n={n}"
                        if n_sup_400 > 0:
                            txt += f"\n{n_sup_400}↑"
                        if n_inf_m200 > 0:
                            txt += f"\n{n_inf_m200}↓"
                        ax.text(x_pos, y_n, txt, ha='center', va='top', fontsize='x-small', rotation=90, color='black')
            # Ajout des labels d'échelle juste au-dessus de l'axe x
            echelle_labels = {'quotidien': 'J', 'quotidien_reduce': 'J*', 'horaire': 'H'}
            for i, s in enumerate(saisons_ligne):
                if s == '__VIDE__':
                    continue
                for j, e in enumerate(echelles):
                    x_pos = i*len(echelles) + j
                    label = echelle_labels.get(e, e)
                    # Positionne le texte juste au-dessus de l'axe x
                    ax.text(x_pos, ymin_ligne + 0.03 * (ymax_ligne - ymin_ligne), label, ha='center', va='bottom', fontsize='medium', fontweight='bold')
            # Adapter les labels x pour n'afficher que la saison sous chaque triplet
            xticks = [i*len(echelles) + 1 for i in range(len(saisons_ligne))]
            xticklabels = [s.upper() if s != '__VIDE__' else '' for s in saisons_ligne]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0, ha='center')
            legend_texte = "Tendance relative" if "z_T_p" in col_calculate else col_calculate
            legend = "%" if "z_T_p" in col_calculate else ""
            ax.set_xlabel("")
            ax.set_ylabel(f"{legend_texte} ({legend})")

        fig.tight_layout(h_pad=1.0)
        fig.savefig(dir_path / f"violin{suffix}.svg", bbox_inches="tight")
        fig.savefig(dir_path / f"violin{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info(dir_path / f"violin{suffix}.svg")

    else:
        logger.warning("Pas de données pour le violin plot.")


def main(args):
    global logger
    logger = get_logger(__name__)

    data_type = args.data_type
    col_calculate = args.col_calculate
    echelle = ["quotidien", "quotidien_reduce", "horaire"]
    season = args.season

    config_mod = load_config("config/modelised_settings.yaml")
    config_obs = load_config("config/observed_settings.yaml")

    datasets = {e: [] for e in echelle}

    for e in echelle:

        for s in season:         
            
            scale = "mm_j" if e == "quotidien" else "mm_h"
            reduce_activate = True if e == "quotidien_reduce" else False

            if data_type == "gev":
                res = import_data_gev(
                    season=s,
                    echelle=e,
                    col_calculate=col_calculate,
                    reduce_activate=reduce_activate,
                    config_mod=config_mod,
                    config_obs=config_obs,
                    scale=scale
                )
                
            logger.info(f"Echelle {e} - Saison {s} données générées")
            datasets[e].append(res)    # On range le résultat dans la bonne liste
    
    generate_violin(
        datasets=datasets,
        dir_path="outputs/boxplot/",
        col_calculate=col_calculate,
        echelle=echelle,
        show_signif=True
    )


def str2bool(v):
    if v == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de génération des représentation")
    parser.add_argument("--data_type", choices=["gev"], default="gev")
    parser.add_argument("--col_calculate", choices=["z_T_p"], default="z_T_p")
    parser.add_argument("--season", type=str, nargs='+', default=[
        "hydro",
        "son",
        "djf",
        "mam",
        "jja",
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
    ])

    args = parser.parse_args()  
    main(args)
