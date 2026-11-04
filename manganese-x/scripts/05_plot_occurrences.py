"""Map the usable Balaghat positives over the district boundary.
Run: python scripts/05_plot_occurrences.py   -> outputs/maps/occurrences.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import load_config, resolve

cfg = load_config()
b = gpd.read_file(resolve(cfg["study_area"]["boundary_file"]))
pos = pd.read_csv(resolve("data/interim/occurrences_mrds_usable.csv"))
pos = pos[pos["in_district"]]
raw = pd.read_csv(resolve("data/raw/occurrences/mrds.csv"), low_memory=False,
                  encoding="latin-1", usecols=["dep_id", "dev_stat", "site_name"])
pos = pos.merge(raw, left_on="occurrence_id", right_on="dep_id", how="left")

fig, ax = plt.subplots(figsize=(8, 9))
b.boundary.plot(ax=ax, color="black", linewidth=1)
colors = {"Occurrence": "tab:blue", "Producer": "tab:red", "Past Producer": "tab:orange"}
for stat, c in colors.items():
    s = pos[pos["dev_stat"] == stat]
    ax.scatter(s["longitude"], s["latitude"], c=c, s=28, label=f"{stat} ({len(s)})",
               edgecolor="white", linewidth=0.5)
ax.set_title("Balaghat: known manganese records (USGS MRDS)\nSource data; not model output")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend()
out = resolve("outputs/maps/occurrences.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)