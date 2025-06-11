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

    print("Téléchargement des reliefs...")
    snapshot_download(
        repo_id="ncsdecoopman/ExtremePrecipit",
        repo_type="dataset",
        revision="main",
        local_dir="data",
        cache_dir=cache_path,
        allow_patterns=["external/*"]
    )
     
    for echelle in ["quotidien", "horaire"]: # , "w3", "w6", "w9", "w12", "w24"
        print(f"Téléchargement des statistiques AROMES (mod)... - Echelle {echelle}")
        snapshot_download(
            repo_id="ncsdecoopman/ExtremePrecipit",
            repo_type="dataset",
            revision="main",
            local_dir="data",
            cache_dir=cache_path,
            allow_patterns=["statisticals/modelised*"]
        )

        print(f"Téléchargement des statistiques STATIONS observées... - Echelle {echelle}")
        snapshot_download(
            repo_id="ncsdecoopman/ExtremePrecipit",
            repo_type="dataset",
            revision="main",
            local_dir="data",
            cache_dir=cache_path,
            allow_patterns=["statisticals/observed*"]
        )   

    # print("Téléchargement des GEVs AROME...")
    # snapshot_download(
    #     repo_id="ncsdecoopman/ExtremePrecipit",
    #     repo_type="dataset",
    #     revision="main",
    #     local_dir="data",
    #     cache_dir=cache_path,
    #     allow_patterns=["gev/modelised*"]
    # )

    # print("Téléchargement des GEVs STATIONS...")
    # snapshot_download(
    #     repo_id="ncsdecoopman/ExtremePrecipit",
    #     repo_type="dataset",
    #     revision="main",
    #     local_dir="data",
    #     cache_dir=cache_path,
    #     allow_patterns=["gev/observed*"]
    # )

except Exception as e:
    print("Erreur pendant le téléchargement :")
    traceback.print_exc()
    raise SystemExit(1)
