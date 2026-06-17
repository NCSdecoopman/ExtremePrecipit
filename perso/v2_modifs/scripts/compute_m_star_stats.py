import pandas as pd
import glob

print("Computing statistics for daily data (hydro year)...")

# OBSERVED (Stations)
obs_best_model_path = r"c:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\gev\gev\observed\quotidien\hydro\gev_param_best_model.parquet"
try:
    df_obs = pd.read_parquet(obs_best_model_path)
    if 'model' in df_obs.columns:
        counts = df_obs['model'].value_counts()
        total = len(df_obs)
        print("Observed total:", total)
        print("Observed model counts:")
        print(counts)
except Exception as e:
    print(f"Error reading observed: {e}")

# MODELISED (Grid points)
mod_best_model_path = r"c:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\gev\gev\modelised\quotidien\hydro\gev_param_best_model.parquet"
try:
    df_mod = pd.read_parquet(mod_best_model_path)
    if 'model' in df_mod.columns:
        counts_mod = df_mod['model'].value_counts()
        total_mod = len(df_mod)
        print("Modelised total:", total_mod)
        print("Modelised model counts:")
        print(counts_mod)
except Exception as e:
    print(f"Error reading modelised: {e}")
