import argparse
import json
from pathlib import Path
import pandas as pd
import xarray as xr
import numcodecs

from pyproj import Transformer
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config_tools import load_config
from src.utils.decode_encode import decode_var, encode_var

def aggregate_metadata(GRID_CSV, MATCH_CSV, N):
    """
    Builds, for each AROME grid point, the following lists:
    - neighboring grid points in an nxn square (excluding center),
    - observation stations (NUM_POSTE_obs) located within this square.
    Independent of order or numeric value of NUM_POSTE.
    """
    # 1. Read input
    grid = pd.read_csv(GRID_CSV/"postes_horaire.csv", dtype={'NUM_POSTE': str})
    match = pd.read_csv(MATCH_CSV/"obs_vs_mod_horaire.csv", dtype={'NUM_POSTE_mod': str, 'NUM_POSTE_obs': str})

    # 2. Grid step
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True) # coordinate projection
    # Project lat/lon from degrees to km
    grid["x"], grid["y"] = transformer.transform(grid["lon"].values, grid["lat"].values)
    match["x_mod"], match["y_mod"] = transformer.transform(match["lon_mod"].values, match["lat_mod"].values)

    # Parameters in meters
    step = 2500         # 2.5 km # Step between AROME grid points
    TOL = 0.5 * step    # 5% margin

    # 3. Stations grouped by grid
    d = (N // 2)*step   # distance (meters) to add/subtract from center to cover NxN square
    expected = N * N - 1 # Expected number of neighbors
    result = []
    fail_neight = 0

    with tqdm(total=len(grid), desc="AROME points") as t:
        for idx, row in grid.iterrows():
            x0, y0, num0 = row['x'], row['y'], row['NUM_POSTE']

            # Project spatial window in meters
            xmin, xmax = x0 - d - TOL, x0 + d + TOL
            ymin, ymax = y0 - d - TOL, y0 + d + TOL

            mask = (
                (grid["x"].between(xmin, xmax)) &
                (grid["y"].between(ymin, ymax))
            )

            square_posts = grid.loc[mask, "NUM_POSTE"].tolist()
            neighbors = [p for p in square_posts if p != num0]

            n_neighbors = len(neighbors) # Number of neighbors found
            if n_neighbors != expected:
                fail_neight += 1

            # Stations associated with all NUM_POSTE_mod in this square
            stations = []
            for p in square_posts: # iterate through all AROME NUM_POSTE identifiers
                stations.extend(match.loc[match["NUM_POSTE_mod"] == p, "NUM_POSTE_obs"].tolist())

            result.append({
                "NUM_POSTE": num0,
                "neighbors_AROME": neighbors,
                "stations_present": stations
            })

            if idx % 10_000 == 0:
                t.update(10_000)

    return result, fail_neight


def process_year(year, centres, voisins_arome, stations_lookup, src_mod_path, src_obs_path,
                 out_mod_path, out_obs_path, var_name, var_conf_mod, var_conf_obs, logger):

    open_kwargs = dict(chunks="auto", decode_cf=False, consolidated=False)
    ds_mod = xr.open_zarr(src_mod_path/f"{year}.zarr", **open_kwargs)
    ds_obs = xr.open_zarr(src_obs_path/f"{year}.zarr", **open_kwargs)

    # Force NUM_POSTE to int (useful if observations use <U8)
    ds_mod = ds_mod.assign_coords(NUM_POSTE=ds_mod.NUM_POSTE.astype(int))
    ds_obs = ds_obs.assign_coords(NUM_POSTE=ds_obs.NUM_POSTE.astype(int))
    pr_mod = decode_var(ds_mod[var_name], var_conf_mod)
    pr_obs = decode_var(ds_obs[var_name], var_conf_obs)

    # Build aggregated tables (calculate in list then concatenate)
    agg_mod_list = []
    agg_obs_list = []

    logger.info("Aggregating...")
    for centre in centres:
        # AROME average (center + neighbors)
        idx_mod = [centre] + voisins_arome.get(centre, [])
        arr_mod = pr_mod.sel(NUM_POSTE=idx_mod).mean("NUM_POSTE", skipna=True)

        # station average with ">= 2 values" constraint, otherwise NaN
        idx_obs = [int(x) for x in stations_lookup[centre]] # station-id ≡ NUM_POSTE
        sel_obs = pr_obs.sel(NUM_POSTE=idx_obs)
        count   = sel_obs.notnull().sum("NUM_POSTE")
        mean    = sel_obs.mean("NUM_POSTE", skipna=True)
        arr_obs = mean.where(count >= 2)                    # NaN if < 2 values

        # Add new dimension "NUM_POSTE" of length 1
        agg_mod_list.append(arr_mod.expand_dims(NUM_POSTE=[centre]))
        agg_obs_list.append(arr_obs.expand_dims(NUM_POSTE=[centre]))

    logger.info("Concatenating...")
    # Concatenation and save
    agg_mod_da = xr.concat(agg_mod_list, dim="NUM_POSTE").sortby("NUM_POSTE")
    agg_obs_da = xr.concat(agg_obs_list, dim="NUM_POSTE").sortby("NUM_POSTE")

    # → convert to Dataset
    agg_mod = agg_mod_da.to_dataset(name=var_name)
    agg_obs = agg_obs_da.to_dataset(name=var_name)

    # Re-encode NaN as fill_value and float as int
    agg_mod[var_name] = encode_var(agg_mod[var_name], var_conf_mod)
    agg_obs[var_name] = encode_var(agg_obs[var_name], var_conf_obs)

    # Keep same "time" coords as input
    logger.info("Applying encoding")
    codec = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=2)
    chunks_tuple = tuple(c[0] for c in agg_mod[var_name].chunks)
    enc = {var_name: {"compressor": codec, "chunks": chunks_tuple}}

    # Write Zarr
    agg_mod.to_zarr(out_mod_path/f"{year}.zarr", mode="w", encoding=enc)
    agg_obs.to_zarr(out_obs_path/f"{year}.zarr", mode="w", encoding=enc)



def pipeline_aggregate(config_obs, config_mod, args):
    global logger
    logger = get_logger(__name__)
    
    config_obs = load_config(args.config_obs)
    config_mod = load_config(args.config_mod)
    N = args.n_aggregate

    # STEP 1 : metadata
    GRID_CSV = Path(config_mod["metadata"]["path"]["outputdir"])
    MATCH_CSV = Path(config_obs["obs_vs_mod"]["metadata_path"]["outputdir"])
    OUT_METADATA = Path(config_obs["obs_vs_mod"]["metadata_path"]["outputdir"])/f"obs_vs_mod_aggregate_n{N}.json"

    if N < 3 or N % 2 == 0:
        raise ValueError("n must be at least 3 and odd")
    else:
        logger.info(f"[AGGREGATE METADATA] Aggregating on {N} * {N}")

    # Metadata aggregation
    result, fail_neight = aggregate_metadata(GRID_CSV, MATCH_CSV, N)
    logger.warning(f"[WARN NEIGHBORS] {fail_neight} AROME grid points with fewer than {N * N - 1} neighbors")

    # JSON writing
    with OUT_METADATA.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON created: {OUT_METADATA.resolve()}")

    # STEP 2 : zarr
    ZARR_OBS_DIR = config_obs["zarr"]["path"]["outputdir"]
    ZARR_MOD_DIR = config_mod["zarr"]["path"]["outputdir"]

    out_mod_path   = Path(ZARR_MOD_DIR) / f"horaire_aggregate_n{N}"
    out_obs_path   = Path(ZARR_OBS_DIR) / f"horaire_aggregate_n{N}"
    src_mod_path   = Path(ZARR_MOD_DIR) / "horaire"
    src_obs_path   = Path(ZARR_OBS_DIR) / "horaire"

    var_name = "pr"
    var_conf_mod = config_mod["zarr"]["variables"][var_name]
    var_conf_obs = config_obs["zarr"]["variables"][var_name]

    # Re-reading metadata
    with open(OUT_METADATA, "r") as f:
        mapping = json.load(f)

    # Dictionaries {center: [neighbors]} and {center: [stations]}
    voisins_arome   = {
        int(m["NUM_POSTE"]): list(map(int, m["neighbors_AROME"]))
        for m in mapping
    }

    # Filter centers
    stations_lookup = {
        int(m["NUM_POSTE"]): m["stations_present"]
        for m in mapping
        # if len(m["stations_present"]) >= 2 # if we want to filter to save time
    }

    # valid centers list
    centres = list(stations_lookup.keys())

    # Extract years from filenames
    available_years_mod = sorted([
        int(p.stem)
        for p in src_mod_path.glob("*.zarr")
        if p.stem.isdigit()
    ])
    available_years_obs = sorted([
        int(p.stem)
        for p in src_obs_path.glob("*.zarr")
        if p.stem.isdigit()
    ])
    assert available_years_mod == available_years_obs, "Mismatch between mod and obs years"


    # STEP 3 : process years
    # from concurrent.futures import ProcessPoolExecutor
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = []
    #     for year in available_years_mod:
    #         futures.append(executor.submit(
    #             process_year,
    # ...
    #         ))

    #     for f in tqdm(futures, desc="Parallel processing of years"):
    #         f.result()  # Raise errors if any
    for year in tqdm(available_years_mod, desc="Processing years"):
        process_year(
            year,
            centres,
            voisins_arome,
            stations_lookup,
            src_mod_path,
            src_obs_path,
            out_mod_path,
            out_obs_path,
            var_name,
            var_conf_mod,
            var_conf_obs,
            logger
        )
        logger.info(f"Aggregated Zarr written:\n  - {out_mod_path}/{year}.zarr - {out_obs_path}/{year}.zarr")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline obs vs mod")
    parser.add_argument("--config_obs", type=str, default="config/observed_settings.yaml")
    parser.add_argument("--config_mod", type=str, default="config/modelised_settings.yaml")
    parser.add_argument("--n_aggregate", type=int, default=3)
    args = parser.parse_args()

    config_obs = load_config(args.config_obs)
    config_mod = load_config(args.config_mod)
    config_obs["echelles"] = "horaire"

    pipeline_aggregate(config_obs, config_mod, args)
