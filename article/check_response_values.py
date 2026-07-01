import pandas as pd

months = ["jan", "fev", "mar", "avr", "mai", "jui", "juill", "aou", "sep", "oct", "nov", "dec"]
h = pd.read_csv("../outputs_nr10/maps/gev_z_T_p/horaire/compare_12/sat_90.0/metrics_signif.csv")
d = pd.read_csv("../outputs/maps/gev_z_T_p/quotidien/compare_12/sat_99.0/metrics_signif.csv")
h["season"] = h.season.str.lower()
d["season"] = d.season.str.lower()

obs_d = "median_obs" if "median_obs" in d.columns else "mean_obs"
mod_d = "median_mod" if "median_mod" in d.columns else "mean_mod"

print("hourly ME panel b")
for m in months:
    r = h[h.season == m].iloc[0]
    print(f"  {m:5} me={r.me:+.2f} r={r.r:.2f} n={int(r.n)}")

print("\ndaily ME panel b")
for m in months:
    r = d[d.season == m].iloc[0]
    print(f"  {m:5} me={r.me:+.2f} r={r.r:.2f}")

print("\nhourly median panel a")
for m in months:
    r = h[h.season == m].iloc[0]
    print(f"  {m:5} obs={r.median_obs:+.2f} mod={r.median_mod:+.2f}")

print("\ndaily median panel a")
for m in months:
    r = d[d.season == m].iloc[0]
    print(f"  {m:5} obs={r[obs_d]:+.2f} mod={r[mod_d]:+.2f}")

d_obs = d.set_index("season")[obs_d].reindex(months)
print("\ndaily median min/max", d_obs.min(), d_obs.idxmin(), d_obs.max(), d_obs.idxmax())
