import polars as pl
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_metrics(df: pl.DataFrame, x_label: str = "AROME", y_label: str = "Station"):
    x = df[x_label].to_numpy()
    y = df[y_label].to_numpy()

    if len(x) != len(y):
        print("Longueur x et y différente")
        return np.nan, np.nan, np.nan, np.nan

    mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) == 0:
        print("Aucune donnée valide après suppression des NaN.")
        return np.nan, np.nan, np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(y_valid, x_valid))
    mae = mean_absolute_error(y_valid, x_valid)
    me = np.mean(x_valid - y_valid)
    corr = np.corrcoef(x_valid, y_valid)[0, 1] if len(x_valid) > 1 else np.nan
    r2_corr = corr**2 if not np.isnan(corr) else np.nan

    return me, mae, rmse, r2_corr, corr

def main():
    pas_de_temps = ["horaire", "quotidien"]
    mapping_files = {
        "horaire": "data/metadonnees/obs_vs_mod/obs_vs_mod_horaire.csv",
        "quotidien": "data/metadonnees/obs_vs_mod/obs_vs_mod_quotidien.csv"
    }

    saison = "djf"
    dfs = {}
    mappings = {}
    for pas in pas_de_temps:
        mapping = pl.read_csv(mapping_files[pas])
        col_obs = "NUM_POSTE_obs"
        col_mod = "NUM_POSTE_mod"
        obs_dir = f"data/gev/observed/{pas}/"
        mod_dir = f"data/gev/modelised/{pas}/"
        obs_file = os.path.join(obs_dir, saison, "niveau_retour.parquet")
        mod_file = os.path.join(mod_dir, saison, "niveau_retour.parquet")
        try:
            df_obs = pl.read_parquet(obs_file)
            df_mod = pl.read_parquet(mod_file)
        except Exception as e:
            print(f"Erreur de chargement pour la saison {saison} ({pas}): {e}")
            continue
        df_obs = df_obs.filter(pl.col("significant") == True)
        df_mod = df_mod.filter(pl.col("significant") == True)
        df_obs = df_obs.rename({"NUM_POSTE": col_obs, "z_T_p": "Station"})
        df_mod = df_mod.rename({"NUM_POSTE": col_mod, "z_T_p": "AROME"})
        mapping = mapping.with_columns([
            pl.col(col_obs).cast(pl.Utf8),
            pl.col(col_mod).cast(pl.Utf8)
        ])
        df_obs = df_obs.with_columns([
            pl.col(col_obs).cast(pl.Utf8)
        ])
        df_mod = df_mod.with_columns([
            pl.col(col_mod).cast(pl.Utf8)
        ])
        df = mapping.join(df_obs, on=col_obs, how="inner").join(df_mod, on=col_mod, how="inner")
        dfs[pas] = df
        mappings[pas] = mapping

    # Trouver les NUM_POSTE communs
    if all(pas in dfs for pas in pas_de_temps):
        num_poste_horaire = set(dfs["horaire"]["NUM_POSTE_obs"].to_list())
        num_poste_quotidien = set(dfs["quotidien"]["NUM_POSTE_obs"].to_list())
        num_poste_communs = sorted(list(num_poste_horaire & num_poste_quotidien))
        # Correction de la génération de la colormap
        color_map = plt.get_cmap('tab20')
        colors = [color_map(i % color_map.N) for i in range(len(num_poste_communs))]
        color_dict = {num: colors[i] for i, num in enumerate(num_poste_communs)}
        default_color = (0.7, 0.7, 0.7, 0.5)  # gris

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Calculer les min/max globaux pour x et y
        all_x = np.concatenate([dfs[pas]["AROME"].to_numpy() for pas in pas_de_temps])
        all_y = np.concatenate([dfs[pas]["Station"].to_numpy() for pas in pas_de_temps])
        min_val = min(np.nanmin(all_x), np.nanmin(all_y))
        max_val = max(np.nanmax(all_x), np.nanmax(all_y))

        for idx, pas in enumerate(pas_de_temps):
            df = dfs[pas]
            x = df["AROME"].to_numpy()
            y = df["Station"].to_numpy()
            num_poste = df["NUM_POSTE_obs"].to_list()
            colors = [color_dict[n] if n in color_dict else default_color for n in num_poste]
            axes[idx].scatter(x, y, alpha=0.7, label="Stations", c=colors)
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
            axes[idx].set_xlabel("AROME")
            axes[idx].set_ylabel("Station")
            axes[idx].set_title(f"Scatter {pas} - {saison}")
            axes[idx].set_xlim(min_val, max_val)
            axes[idx].set_ylim(min_val, max_val)
            axes[idx].legend()
        # Créer une légende personnalisée pour les NUM_POSTE communs
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=color_dict[n], label=n) for n in num_poste_communs]
        axes[1].legend(handles=legend_patches, title="NUM_POSTE communs", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        print("Impossible de charger les deux pas de temps pour la saison demandée.")

if __name__ == "__main__":
    main() 