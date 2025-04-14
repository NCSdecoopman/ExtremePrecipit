# deploy/download_data.py

from huggingface_hub import snapshot_download

print("Téléchargement des fichiers GEV...")
snapshot_download(
    repo_id="ncsdecoopman/ExtremePrecipit",
    repo_type="dataset",
    revision="main",
    local_dir="data/gev",
    cache_dir="/tmp/hf_cache",
    allow_patterns=["gev/*"]
)

print("Téléchargement des métadonnées...")
snapshot_download(
    repo_id="ncsdecoopman/ExtremePrecipit",
    repo_type="dataset",
    revision="main",
    local_dir="data/metadonnees",
    cache_dir="/tmp/hf_cache",
    allow_patterns=["metadonnees/*"]
)

print("Téléchargement des statistiques...")
snapshot_download(
    repo_id="ncsdecoopman/ExtremePrecipit",
    repo_type="dataset",
    revision="main",
    local_dir="data/statisticals",
    cache_dir="/tmp/hf_cache",
    allow_patterns=["statisticals/*"]
)
