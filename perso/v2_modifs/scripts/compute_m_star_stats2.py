import pandas as pd
import numpy as np
from scipy.stats import chi2
import os

def compute_stats(obs_or_mod, data_base_dir, seasons):
    results = {}
    for season in seasons:
        data_dir = os.path.join(data_base_dir, season)
        s_gev_path = os.path.join(data_dir, "gev_param_s_gev.parquet")
        best_path = os.path.join(data_dir, "gev_param_best_model.parquet")
        
        if not os.path.exists(s_gev_path) or not os.path.exists(best_path):
            results[season] = None
            continue
            
        try:
            df_s = pd.read_parquet(s_gev_path)[['NUM_POSTE', 'log_likelihood']].rename(columns={'log_likelihood': 'll_s'})
            df_best = pd.read_parquet(best_path)[['NUM_POSTE', 'model', 'log_likelihood']].rename(columns={'log_likelihood': 'll_ns'})
            
            df = pd.merge(df_s, df_best, on='NUM_POSTE')
            
            df['k'] = df['model'].apply(lambda x: 2 if 'm3' in str(x).lower() else 1)
            df['LRT'] = 2 * (df['ll_ns'] - df['ll_s'])
            df['pval'] = chi2.sf(df['LRT'], df['k'])
            
            df['final_model'] = df.apply(lambda row: row['model'] if row['pval'] <= 0.10 else 'M0', axis=1)
            
            total = len(df)
            m_star_count = sum('break' in str(x).lower() or '*' in str(x) for x in df['final_model'])
            
            m_star_pct = m_star_count / total * 100
            results[season] = m_star_pct
            
        except Exception as e:
            print(f"Error for {obs_or_mod} - {season}: {e}")
            results[season] = None
            
    return results

obs_base = r"c:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\gev\gev\observed\quotidien"
mod_base = r"c:\Users\nicod\Documents\GitHub\ExtremePrecipit\data\gev\gev\modelised\quotidien"
seasons = ["hydro", "ond", "jfm", "amj", "jas"]

obs_res = compute_stats("Observed (Stations)", obs_base, seasons)
mod_res = compute_stats("Modelised (Grid Points)", mod_base, seasons)

print("--- Observed (Stations) M* Selection % ---")
for s in seasons:
    if obs_res[s] is not None:
        print(f"{s.upper()}: {obs_res[s]:.1f}%")

print("\n--- Modelised (Grid Points) M* Selection % ---")
for s in seasons:
    if mod_res[s] is not None:
        print(f"{s.upper()}: {mod_res[s]:.1f}%")
