import polars as pl
import numpy as np
from pathlib import Path

break_year = 1985

def compute_zT_for_years_VERSION(
    annees_retour: np.ndarray,  # tableau d'années souhaitées
    min_year: int,
    max_year: int,
    df_params: pl.DataFrame,
    df_series: pl.DataFrame,
    T=10
) -> pl.DataFrame:
    """
    Calcule z_T(x) pour chaque station pour deux années de retour (a et b),
    et retourne un DataFrame avec NUM_POSTE, zTpa, zTpb, (zTpb-zTpa)/zTpa.
    """
    rows = []
    for row in df_params.to_dicts():
        data_station = df_series.filter(pl.col("NUM_POSTE") == row["NUM_POSTE"])
        years_obs = data_station["year"].to_numpy()
        t_tilde_obs_raw = (years_obs - min_year) / (max_year - min_year)
        t_min_ret = t_tilde_obs_raw.min()
        t_max_ret = t_tilde_obs_raw.max()
        res0_obs = t_tilde_obs_raw / (t_max_ret - t_min_ret)
        dx = res0_obs.min() + 0.5

        t_tilde_retour = []
        model = row["model"]
        break_model = "_break" in model
        
        for y in annees_retour:
            if break_model and y<=1985:
                t_tilde = 0
            else:
                t_tilde_retour_raw = (y - min_year) / (max_year - min_year)
                res0_ret = t_tilde_retour_raw / (t_max_ret - t_min_ret)
                t_tilde = res0_ret - dx

            t_tilde_retour.append(t_tilde)
        t_tilde_retour = np.array(t_tilde_retour)



        mu0, mu1, sigma0, sigma1, xi = row["mu0"], row["mu1"], row["sigma0"], row["sigma1"], row["xi"]
        CT = ((-np.log(1-1/T))**(-xi) - 1)
        zT = mu0 + mu1*t_tilde_retour + (sigma0 + sigma1*t_tilde_retour)/xi * CT

        rows.append({
            "NUM_POSTE": row["NUM_POSTE"],
            "zT": zT
        })

    return pl.DataFrame(rows)


def plot_series_station(station_id):
    import matplotlib.pyplot as plt
    from src.utils.data_utils import cleaning_data_observed, load_data
    from src.pipelines.pipeline_best_to_niveau_retour import build_x_ttilde
    break_year = 1985

    # Fichiers Parquet GEV
    file_1990_2022 = Path('data/gev/observed/horaire/nov/gev_param_best_model.parquet')
    file_1959_2022 = Path('data/gev/observed/quotidien/nov/gev_param_best_model.parquet')
    gev_1959_2022 = pl.read_parquet(file_1959_2022)
    gev_1990_2022 = pl.read_parquet(file_1990_2022)

    # Paramètre de chargement des données
    input_dir_quot = Path("data/statisticals/observed/quotidien")
    input_dir_hor = Path("data/statisticals/observed/horaire")
    mesure_quot = "max_mm_j"
    mesure_hor = "max_mm_h"
    cols_quot = ["NUM_POSTE", mesure_quot, "nan_ratio"]
    cols_hor = ["NUM_POSTE", mesure_hor, "nan_ratio"]
    df_1959_2022 = load_data(input_dir_quot, "nov", "quotidien", cols_quot, 1959, 2022)
    df_1990_2022 = load_data(input_dir_hor, "nov", "horaire", cols_hor, 1990, 2022)

    df_1959_2022 = cleaning_data_observed(df_1959_2022, "quotidien", len_serie=50)
    df_1990_2022 = cleaning_data_observed(df_1990_2022, "horaire", len_serie=25)
    df_1959_2022 = df_1959_2022.drop_nulls(subset=[mesure_quot])
    df_1990_2022 = df_1990_2022.drop_nulls(subset=[mesure_hor])

    df_series_1959_2022 = build_x_ttilde(df_1959_2022, 1959, 2022, gev_1959_2022, break_year, mesure_quot)
    df_series_1990_2022 = build_x_ttilde(df_1990_2022, 1990, 2022, gev_1990_2022, break_year, mesure_hor)

    z_levels_1959_2022 = compute_zT_for_years_VERSION(
        np.arange(1959, 2023),
        1959,
        2022,
        gev_1959_2022,
        df_series_1959_2022
    )
    z_levels_1990_2022 = compute_zT_for_years_VERSION(
        np.arange(1990, 2023),
        1990,
        2022,
        gev_1990_2022,
        df_series_1990_2022
    )

    # Ajout de la série 'quotidien_reduce' (1990-2022)
    file_quot_reduce = Path('data/gev/observed/quotidien_reduce/nov/gev_param_best_model.parquet')
    gev_quot_reduce = pl.read_parquet(file_quot_reduce)
    input_dir_quot_reduce = Path("data/statisticals/observed/quotidien")
    mesure_quot_reduce = "max_mm_j"
    cols_quot_reduce = ["NUM_POSTE", mesure_quot_reduce, "nan_ratio"]
    df_quot_reduce = load_data(input_dir_quot_reduce, "nov", "quotidien_reduce", cols_quot_reduce, 1990, 2022)
    df_quot_reduce = cleaning_data_observed(df_quot_reduce, "quotidien", len_serie=25)
    df_quot_reduce = df_quot_reduce.drop_nulls(subset=[mesure_quot_reduce])
    df_series_quot_reduce = build_x_ttilde(df_quot_reduce, 1990, 2022, gev_quot_reduce, break_year, mesure_quot_reduce)
    z_levels_quot_reduce = compute_zT_for_years_VERSION(
        np.arange(1990, 2023),
        1990,
        2022,
        gev_quot_reduce,
        df_series_quot_reduce
    )
    # Série quotidienne classique mais restreinte à 1990-2022
    cols_quot_1990 = ["NUM_POSTE", mesure_quot, "nan_ratio"]
    df_quot_1990_2022 = load_data(input_dir_quot, "nov", "quotidien", cols_quot_1990, 1990, 2022)
    df_quot_1990_2022 = cleaning_data_observed(df_quot_1990_2022, "quotidien", len_serie=25)
    df_quot_1990_2022 = df_quot_1990_2022.drop_nulls(subset=[mesure_quot])

    # Extraction pour la station
    station_id_cast = station_id if isinstance(station_id, str) else str(station_id)
    df_station = df_1959_2022.filter(pl.col("NUM_POSTE") == station_id_cast)
    print("Données journalières (1959-2022)")
    print(gev_1959_2022.filter(pl.col("NUM_POSTE") == station_id_cast))
    years = df_station["year"].to_numpy()
    values = df_station[mesure_quot].to_numpy()
    df_station_hor = df_1990_2022.filter(pl.col("NUM_POSTE") == station_id_cast)
    print("Données horaires (1990-2022)")
    print(gev_1990_2022.filter(pl.col("NUM_POSTE") == station_id_cast))
    years_hor = df_station_hor["year"].to_numpy()
    values_hor = df_station_hor[mesure_hor].to_numpy()
    z_1959_2022 = z_levels_1959_2022.filter(pl.col("NUM_POSTE") == station_id_cast)["zT"].to_numpy()[0]
    z_1990_2022 = z_levels_1990_2022.filter(pl.col("NUM_POSTE") == station_id_cast)["zT"].to_numpy()[0]
    z_quot_reduce = z_levels_quot_reduce.filter(pl.col("NUM_POSTE") == station_id_cast)["zT"].to_numpy()[0]
    years_1959_2022 = np.arange(1959, 2023)
    years_1990_2022 = np.arange(1990, 2023)
    years_quot_reduce_grid = np.arange(1990, 2023)
    df_station_quot_1990_2022 = df_quot_1990_2022.filter(pl.col("NUM_POSTE") == station_id_cast)
    print("Données journalières (1990-2022)")
    print(gev_quot_reduce.filter(pl.col("NUM_POSTE") == station_id_cast))
    years_quot_1990_2022 = df_station_quot_1990_2022["year"].to_numpy()
    values_quot_1990_2022 = df_station_quot_1990_2022[mesure_quot].to_numpy()

    # Calcul des tendances
    zT_1992_1 = z_1959_2022[1992 - 1959]
    zT_2022_1 = z_1959_2022[2022 - 1959]
    trend_1959 = ((zT_2022_1 - zT_1992_1) / zT_1992_1) * 100
    zT_1992_2 = z_1990_2022[1992 - 1990]
    zT_2022_2 = z_1990_2022[2022 - 1990]
    trend_1990 = ((zT_2022_2 - zT_1992_2) / zT_1992_2) * 100
    zT_1992_reduce = z_quot_reduce[1992 - 1990]
    zT_2022_reduce = z_quot_reduce[2022 - 1990]
    trend_reduce = ((zT_2022_reduce - zT_1992_reduce) / zT_1992_reduce) * 100

    # Création des trois facettes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5*3, 7.5), sharex=True)

    # --- Facette 1 : Données journalières (1959-2022) ---
    ax1.plot(years, values, label="Série observée quotidienne", color="black", marker="o", linestyle="-", alpha=0.5)
    ax1.plot(years_1959_2022, z_1959_2022, label="Niveau de retour 1959-2022", color="blue")
    ax1.set_ylabel("Précipitation (mm/j)")
    ax1.set_title("Données journalières (1959-2022) en novembre")
    leg1 = ax1.legend(loc="upper left")
    ax1.grid(True)
    for year in [1992, 2022]:
        idx = year - 1959
        value = z_1959_2022[idx]
        ax1.annotate(
            f"{value:.1f}",
            (year, value),
            textcoords="offset points",
            xytext=(0, 10 if year == 1990 else -15),
            ha='center',
            color="blue",
            fontsize=10,
            fontweight='bold',
            bbox=None
        )
    ax1.scatter([1992, 2022], [z_1959_2022[1992-1959], z_1959_2022[-1]], s=40, color='blue', zorder=5)
    ax1.text(
        0.02, 0.88, f"Tendance : {trend_1959:.1f}%", transform=ax1.transAxes,
        color='blue', fontsize=12, fontweight='bold', va='top', ha='left'
    )
    ax1.set_xlabel("Année")

    # --- Facette 2 : Données horaires (1990-2022) ---
    ax3.plot(years_hor, values_hor, label="Série observée horaire", color="black", marker="o", linestyle="-", alpha=0.5)
    ax3.plot(years_1990_2022, z_1990_2022, label="Niveau de retour 1990-2022", color="red")
    ax3.set_ylabel("Précipitation (mm/h)")
    ax3.set_title("Données horaires (1990-2022) en novembre")
    leg3 = ax3.legend(loc="upper left")
    ax3.grid(True)
    for year in [1992, 2022]:
        idx = year - 1990
        value = z_1990_2022[idx]
        ax3.annotate(
            f"{value:.1f}",
            (year, value),
            textcoords="offset points",
            xytext=(0, 10 if year == 1990 else -15),
            ha='center',
            color="red",
            fontsize=10,
            fontweight='bold',
            bbox=None
        )
    ax3.scatter([1992, 2022], [z_1990_2022[1992-1990], z_1990_2022[-1]], s=40, color='red', zorder=5)
    ax3.text(
        0.02, 0.88, f"Tendance : {trend_1990:.1f}%", transform=ax3.transAxes,
        color='red', fontsize=12, fontweight='bold', va='top', ha='left'
    )
    ax3.set_xlabel("Année")

    # --- Facette 3 : Données journalières (1990-2022) ---
    ax2.plot(years_quot_1990_2022, values_quot_1990_2022, label="Série observée quotidienne", color="black", marker="o", linestyle="-", alpha=0.5)
    ax2.plot(years_quot_reduce_grid, z_quot_reduce, label="Niveau de retour 1990-2022", color="orange")
    ax2.set_ylabel("Précipitation (mm/j)")
    ax2.set_title("Données journalières (1990-2022) en novembre")
    leg2 = ax2.legend(loc="upper left")
    ax2.grid(True)
    for year in [1992, 2022]:
        idx = year - 1990
        if 0 <= idx < len(z_quot_reduce):
            value = z_quot_reduce[idx]
            ax2.annotate(
                f"{value:.1f}",
                (year, value),
                textcoords="offset points",
                xytext=(0, 10 if year == 1990 else -15),
                ha='center',
                color="orange",
                fontsize=10,
                fontweight='bold',
                bbox=None
            )
    ax2.scatter([1992, 2022], [z_quot_reduce[1992-1990], z_quot_reduce[-1]], s=40, color='orange', zorder=5)
    ax2.text(
        0.02, 0.88, f"Tendance : {trend_reduce:.1f}%", transform=ax2.transAxes,
        color='orange', fontsize=12, fontweight='bold', va='top', ha='left'
    )
    ax2.set_xlabel("Année")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(bottom=0.12, wspace=0.2)

    # Enregistrement
    output_dir = Path("presentation/schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{station_id}_serie_nov.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    plot_series_station("26313001")
