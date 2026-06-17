import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "app"))
sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
import os
from app.utils.data_utils import match_and_compare, add_metadata

for scale in ['quotidien', 'horaire']:
    print('---', scale, '---')
    val_scale = 'mm_j' if scale == 'quotidien' else 'mm_h'
    for m in ['jan', 'fev', 'mar', 'avr', 'mai', 'jui', 'juill', 'aou', 'sep', 'oct', 'nov', 'dec']:
        p_obs = f'data/gev_m0/observed/{scale}/{m}/niveau_retour.parquet'
        p_mod = f'data/gev_m0/modelised/{scale}/{m}/niveau_retour.parquet'
        if os.path.exists(p_obs) and os.path.exists(p_mod):
            obs = pl.read_parquet(p_obs)
            mod = pl.read_parquet(p_mod)
            
            # Add metadata
            obs = add_metadata(obs, val_scale, type='observed')
            mod = add_metadata(mod, val_scale, type='modelised')
            
            # Cast types
            obs = obs.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
            mod = mod.with_columns(pl.col("NUM_POSTE").cast(pl.Int32))
            
            df_obs_vs_mod = pl.read_csv(f"data/metadonnees/obs_vs_mod/obs_vs_mod_{scale}.csv")
            
            # Significant only
            obs_sig = obs.filter(pl.col("significant") == True)
            
            matched = match_and_compare(obs_sig, mod, "z_T_p", df_obs_vs_mod)
            
            bias_mean = (matched["AROME"] - matched["Station"]).mean()
            bias_med = (matched["AROME"] - matched["Station"]).median()
            
            print(f'{m}: n={matched.shape[0]} | obs_mean={matched["Station"].mean():.2f}, obs_med={matched["Station"].median():.2f} | mod_mean={matched["AROME"].mean():.2f}, mod_med={matched["AROME"].median():.2f} | bias_mean={bias_mean:.2f}, bias_med={bias_med:.2f}')
