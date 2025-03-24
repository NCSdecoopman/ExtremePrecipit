import numpy as np
import pandas as pd
import requests
import time


API_KEY = "eyJ4NXQiOiJZV0kxTTJZNE1qWTNOemsyTkRZeU5XTTRPV014TXpjek1UVmhNbU14T1RSa09ETXlOVEE0Tnc9PSIsImtpZCI6ImdhdGV3YXlfY2VydGlmaWNhdGVfYWxpYXMiLCJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJuaWNvZGVjNTdAY2FyYm9uLnN1cGVyIiwiYXBwbGljYXRpb24iOnsib3duZXIiOiJuaWNvZGVjNTciLCJ0aWVyUXVvdGFUeXBlIjpudWxsLCJ0aWVyIjoiVW5saW1pdGVkIiwibmFtZSI6IkRlZmF1bHRBcHBsaWNhdGlvbiIsImlkIjoyNTg4MSwidXVpZCI6IjkyOGM1ODU3LTFkMzctNGI3Mi1iODQxLThkZDk0ZTJiNTliOSJ9LCJpc3MiOiJodHRwczpcL1wvcG9ydGFpbC1hcGkubWV0ZW9mcmFuY2UuZnI6NDQzXC9vYXV0aDJcL3Rva2VuIiwidGllckluZm8iOnsiNTBQZXJNaW4iOnsidGllclF1b3RhVHlwZSI6InJlcXVlc3RDb3VudCIsImdyYXBoUUxNYXhDb21wbGV4aXR5IjowLCJncmFwaFFMTWF4RGVwdGgiOjAsInN0b3BPblF1b3RhUmVhY2giOnRydWUsInNwaWtlQXJyZXN0TGltaXQiOjAsInNwaWtlQXJyZXN0VW5pdCI6InNlYyJ9fSwia2V5dHlwZSI6IlBST0RVQ1RJT04iLCJzdWJzY3JpYmVkQVBJcyI6W3sic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzQ2xpbWF0b2xvZ2llIiwiY29udGV4dCI6IlwvcHVibGljXC9EUENsaW1cL3YxIiwicHVibGlzaGVyIjoiYWRtaW5fbWYiLCJ2ZXJzaW9uIjoidjEiLCJzdWJzY3JpcHRpb25UaWVyIjoiNTBQZXJNaW4ifSx7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiRG9ubmVlc1B1YmxpcXVlc1BhcXVldE9ic2VydmF0aW9uIiwiY29udGV4dCI6IlwvcHVibGljXC9EUFBhcXVldE9ic1wvdjEiLCJwdWJsaXNoZXIiOiJiYXN0aWVuZyIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzT2JzZXJ2YXRpb24iLCJjb250ZXh0IjoiXC9wdWJsaWNcL0RQT2JzXC92MSIsInB1Ymxpc2hlciI6ImJhc3RpZW5nIiwidmVyc2lvbiI6InYxIiwic3Vic2NyaXB0aW9uVGllciI6IjUwUGVyTWluIn1dLCJleHAiOjE3NDM3MDA5NzcsInRva2VuX3R5cGUiOiJhcGlLZXkiLCJpYXQiOjE3NDI4MzY5NzcsImp0aSI6IjIwMjcyNjIwLTE5MjMtNGUyZS1iYmE4LWIyZmRlMGY2ODQ3ZSJ9.KdCm-X7h71B5ErMtRhvMA2gmcrSoATxgJ5QvD5Vlsk6CHDrqmBD3Uj5RELloN3AHe8DH80TggbCRJWRtnjYgp8enDqz-6JcyZ9CmOe6CMm7E3ehQXrpO5FT67t8ap4wMZD4GrvBIbWbks2b51BUE7FKhB-FNueVTDVBvUEbUW_VAfGcNpR2uVSTNbZDdZ5GnpBupCyL8yG8y2jQQaiPBjKV0b4wz2Im_Mbv6knLMln-9Ivxh8aUey2EcAu7gdK10qCHwcCPpBXNTy87fLhCmtla_E9N9fcj2gDma2qhSBu2OeEGay4UNhtY9E2AhuSt7MYT0fD9f71w-a3m2gBkWXg=="
BASE_URL = "https://public-api.meteofrance.fr/public/DPClim/v1"

def get_total_stations(
        id_departement: int,
        echelle: str, 
        api_key: str,
        parametre: str = 'precipitation'):
    
    url = f"{BASE_URL}/liste-stations/{echelle}"
    headers = {
        'apikey': api_key
    }
    params = {
        'id-departement': id_departement,
        'parametre': parametre
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"Département {id_departement:02d} - {len(data)} stations trouvées à l'échelle {echelle}")
        return data
    else:
        print(f"Erreur {response.status_code}: {response.text}")
        return []

# Récupération de tous les départements
all_stations = []

# Récupération de toutes les stations à enregistrement horaire ou journalière
for echelle in ["horaire", "quotidienne"]:
    for dep in range(1, 96):
        stations = get_total_stations(dep, echelle, API_KEY)
        for s in stations:
            all_stations.append({
                "id": s["id"],
                "lat": np.float32(s["lat"]),
                "lon": np.float32(s["lon"]),
                "nom": s["nom"],
                "echelle": echelle
            })
        time.sleep(1.5)  # pour éviter l'erreur 429

# Transformation en DataFrame
df = pd.DataFrame(all_stations)

# Pivot pour obtenir horaire / quotidienne en booléens
df["horaire"] = df["echelle"] == "horaire"
df["quotidienne"] = df["echelle"] == "quotidienne"

df_final = df.groupby(["id", "lat", "lon", "nom"]).agg({
    "horaire": "max",
    "quotidienne": "max"
}).reset_index()

# Conversion booléenne propre
df_final["horaire"] = df_final["horaire"].astype(bool)
df_final["quotidienne"] = df_final["quotidienne"].astype(bool)

print(df_final.head())

# Export CSV
df_final.to_csv("stations_meteofrance.csv", index=False)
