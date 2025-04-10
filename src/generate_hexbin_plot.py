import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_obs_vs_mod_hexbin_percent(df_parquet_path, echelle, output_path):
    """
    Génère un hexbin plot pr_obs vs pr_mod, coloré en pourcentage du total de points,
    et l'enregistre dans le fichier output_path.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Lire le fichier Parquet
    df = pd.read_parquet(df_parquet_path)

    # Nettoyer les données : supprimer les NaN
    df_clean = df.dropna(subset=["pr_obs", "pr_mod"])
    pr_obs = df_clean["pr_obs"].to_numpy()
    pr_mod = df_clean["pr_mod"].to_numpy()

    # Création du hexbin
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(
        pr_mod, pr_obs,
        gridsize=100,
        cmap='viridis',
        mincnt=1,
        reduce_C_function=np.sum
    )

    # Récupérer les comptes et les transformer en %
    counts = hb.get_array()
    total = counts.sum()
    hb.set_array(100 * counts / total)

    # Diagonale y = x
    min_val = min(pr_obs.min(), pr_mod.min())
    max_val = max(pr_obs.max(), pr_mod.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    # Colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Pourcentage des points (%)")

    # Mise en forme
    plt.xlabel("Modélisation (pr_mod)")
    plt.ylabel("Observation (pr_obs)")
    plt.title(f"Hexbin : pr_obs vs pr_mod ({echelle}, {len(pr_obs):,} points)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Enregistrement
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Hexbin sauvegardé dans : {output_path}")



if __name__ == "__main__":

    plot_obs_vs_mod_hexbin_percent("data/obs_vs_mod/obs_vs_mod_horaire.parquet", 
                                   "horaire",
                                   output_path="data/obs_vs_mod/hexbin_horaire.png")
