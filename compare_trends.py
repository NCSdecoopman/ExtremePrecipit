import pandas as pd
import matplotlib.pyplot as plt

def compare_trends():
    # Paths for full period (1959-2022) and restricted period (1990-2022)
    path_full = "data/gev/modelised/horaire_reduce/hydro/niveau_retour.parquet"
    path_rest = "data/gev/modelised/horaire/hydro/niveau_retour.parquet"

    print("--- Loading data ---")
    try:
        df_full = pd.read_parquet(path_full)
        df_rest = pd.read_parquet(path_rest)
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return

    col = 'z_T_p'
    if col not in df_full.columns or col not in df_rest.columns:
        print(f"Error: Column '{col}' not found in one of the files.")
        print(f"Available in full: {df_full.columns.tolist()}")
        print(f"Available in rest: {df_rest.columns.tolist()}")
        return

    # Filter valid data
    trend_full = df_full[col].dropna()
    trend_rest = df_rest[col].dropna()

    print(f"Full period (1959-2022): {len(trend_full)} stations")
    print(f"Restricted period (1990-2022): {len(trend_rest)} stations")

    # Statistics comparison
    stats = pd.DataFrame({
        "1959-2022 (Full)": [
            trend_full.mean(), 
            trend_full.median(), 
            trend_full.min(), 
            trend_full.max(), 
            (trend_full > 0).mean() * 100
        ],
        "1990-2022 (Restricted)": [
            trend_rest.mean(), 
            trend_rest.median(), 
            trend_rest.min(), 
            trend_rest.max(), 
            (trend_rest > 0).mean() * 100
        ]
    }, index=["Mean", "Median", "Min", "Max", "% Positive Trends"])

    print("\n--- Statistics Summary (z_T_p) ---")
    print(stats.to_string())

    # Range and Strength Comparison
    mean_full = trend_full.mean()
    mean_rest = trend_rest.mean()
    
    print("\n--- Analysis ---")
    if mean_rest > mean_full:
        print(f"Trends are STRONGER in the RESTRICTED period (1990-2022) on average ({mean_rest:.3f} vs {mean_full:.3f}).")
    else:
        print(f"Trends are STRONGER in the FULL period (1959-2022) on average ({mean_full:.3f} vs {mean_rest:.3f}).")

    # Range (Max - Min) comparison
    range_full = trend_full.max() - trend_full.min()
    range_rest = trend_rest.max() - trend_rest.min()
    print(f"Range in Full: {range_full:.3f}, Range in Restricted: {range_rest:.3f}")
    if range_rest > range_full:
        print("The variability (range) is HIGHER in the RESTRICTED period.")
    else:
        print("The variability (range) is HIGHER in the FULL period.")

    # Plotting Boxplot
    plt.figure(figsize=(10, 7))
    plt.boxplot([trend_full, trend_rest], labels=['1959-2022 (Full)', '1990-2022 (Restricted)'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    
    plt.title('Comparison of Trends (z_T_p) for AROME Hourly', fontsize=14)
    plt.ylabel('Trend Value (z_T_p)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    output_png = "trend_comparison_boxplot.png"
    plt.savefig(output_png)
    print(f"\nBoxplot saved as: {output_png}")
    plt.show()

if __name__ == "__main__":
    compare_trends()
