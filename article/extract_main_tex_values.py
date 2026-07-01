import pandas as pd

months = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
hourly = pd.read_csv("outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv")
hourly["season"] = hourly["season"].str.lower()
hourly = hourly.set_index("season").reindex(months).reset_index()

daily_path = "outputs/maps/gev_z_T_p/quotidien/compare_12/sat_99.0/metrics_signif.csv"
daily = pd.read_csv(daily_path)
daily["season"] = daily["season"].str.lower()
daily = daily.set_index("season").reindex(months).reset_index()

obs_col = "median_obs" if "median_obs" in daily.columns else "mean_obs"
mod_col = "median_mod" if "median_mod" in daily.columns else "mean_mod"

dmin = daily.loc[daily[obs_col].idxmin()]
dmax = daily.loc[daily[obs_col].idxmax()]
print("daily_min", dmin["season"], f"{dmin[obs_col]:+.2f}")
print("daily_max", dmax["season"], f"{dmax[obs_col]:+.2f}")

for m in ["fev", "mar", "jui", "oct", "dec"]:
    row = hourly[hourly["season"] == m].iloc[0]
    print(
        m,
        f"obs={row['median_obs']:+.2f}",
        f"mod={row['median_mod']:+.2f}",
        f"me={row['me']:+.1f}",
        f"r={row['r']:.2f}",
    )

for label, path, season in [
    ("daily_ond", "outputs/maps/gev_z_T_p/quotidien/compare_5/sat_99.0/metrics_signif.csv", "ond"),
    ("hourly_hydro", "outputs_nr10/maps/gev_z_T_p/horaire/compare_5/sat_90.0/metrics_signif.csv", "hydro"),
    ("hourly_amj", "outputs_nr10/maps/gev_z_T_p/horaire/compare_5/sat_90.0/metrics_signif.csv", "amj"),
    ("hourly_jas", "outputs_nr10/maps/gev_z_T_p/horaire/compare_5/sat_90.0/metrics_signif.csv", "jas"),
]:
    df = pd.read_csv(path)
    row = df[df["season"].str.lower() == season].iloc[0]
    print(label, f"r={row['r']:.2f}", f"me={row['me']:+.1f}")

row_d = daily[daily["season"] == "jui"].iloc[0]
row_h = hourly[hourly["season"] == "jui"].iloc[0]
print("june", f"daily_me={row_d['me']:+.1f}", f"hourly_me={row_h['me']:+.1f}")

for m in ["jan", "mar"]:
    print(f"daily_me_{m}", f"{daily[daily['season'] == m]['me'].iloc[0]:+.1f}")

print("daily_r_dec", f"{daily[daily['season'] == 'dec']['r'].iloc[0]:.2f}")
for m in ["jui", "sep", "mai", "mar", "oct", "fev"]:
    print(f"hourly_r_{m}", f"{hourly[hourly['season'] == m]['r'].iloc[0]:.2f}")

m = pd.read_csv("outputs_nr10/maps/gev_zTpa/quotidien/compare_1/sat_99.0/metrics.csv")
print("dNRJour", f"{m[m.season.str.lower()=='hydro']['delta'].iloc[0]:+.2f}")
