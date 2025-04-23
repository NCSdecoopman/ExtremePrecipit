from huggingface_hub import snapshot_download
import os
import traceback

cache_path = os.path.expanduser("~/.cache/huggingface/hub")

try:
    print("Téléchargement des métadonnées...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["metadonnees/*"]
    )
    print("Téléchargement des statistiques AROMES (mod)...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["statisticals/modelised*"]
    )
    print("Téléchargement des statistiques STATIONS (obs)...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["statisticals/observed*"]
    )
    print("Téléchargement des GEV AROMS (mod)...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev/modelised*"]
    )
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev_non_sta/modelised*"]
    )
    print("Téléchargement des GEV sta STATIONS (obs)...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev/observed*"]
    )
    print("Téléchargement des GEV non stat STATIONS (obs)...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev_non_sta/observed*"]
    )


except Exception as e:
    print("Erreur pendant le téléchargement :")
    traceback.print_exc()
    raise SystemExit(1)
