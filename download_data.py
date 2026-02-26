from huggingface_hub import snapshot_download
import os
import traceback

cache_path = os.path.expanduser("~/.cache/huggingface/hub")

try:
    print("Downloading metadata...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["metadonnees/*"]
    )

    print("Downloading relief data...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["external/*"]
    )
     
    for echelle in ["quotidien", "horaire", "w3", "w6", "w9", "w12", "w24"]:
        print(f"Downloading AROMES statistics (mod) - Scale {echelle}...")
        snapshot_download(
            repo_id="ncsdecoopman/ExtremePrecipit",
            repo_type="dataset",
            revision="main",
            local_dir="data",
            cache_dir=cache_path,
            allow_patterns=["statisticals/modelised*"]
        )

        print(f"Downloading observed STATION statistics - Scale {echelle}...")
        snapshot_download(
            repo_id="ncsdecoopman/ExtremePrecipit",
            repo_type="dataset",
            revision="main",
            local_dir="data",
            cache_dir=cache_path,
            allow_patterns=["statisticals/observed*"]
        )   

    print("Downloading AROME GEVs...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev/modelised*"]
    )

    print("Downloading STATION GEVs...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["gev/observed*"]
    )

except Exception as e:
    print("Download error:")
    traceback.print_exc()
    raise SystemExit(1)
