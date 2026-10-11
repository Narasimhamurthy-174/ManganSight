"""Look at the usable Balaghat positives: what fields exist, how clustered are they?
Run: python scripts/03_inspect_occurrences.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.common.config import load_config, resolve

cfg = load_config()
pos = pd.read_csv(resolve("data/interim/occurrences_mrds_usable.csv"))
pos = pos[pos["in_district"]].copy()
print(f"Inside-district usable positives: {len(pos)}\n")

# Full MRDS rows for these records (all columns, to see what geology fields exist)
raw = pd.read_csv(resolve("data/raw/occurrences/mrds.csv"), low_memory=False, encoding="latin-1")
raw.columns = [c.lower() for c in raw.columns]
full = raw[raw["dep_id"].isin(pos["occurrence_id"])]
print("ALL MRDS columns:", list(raw.columns), "\n")

filled = full.notna().sum().sort_values(ascending=False)
print("Columns with data for these records (non-null counts):")
print(filled[filled > 0].to_string(), "\n")

for col in ["dev_stat", "com_type", "commod1", "commod2", "commod3", "dep_type", "hrock_unit", "hrock_type"]:
    if col in full.columns:
        print(f"--- {col} ---")
        print(full[col].value_counts(dropna=False).head(10).to_string(), "\n")

# Clustering diagnostics
tf = Transformer.from_crs("EPSG:4326", cfg["crs"]["projected"], always_xy=True)
x, y = tf.transform(pos["longitude"].values, pos["latitude"].values)
xy = np.column_stack([x, y])
d, _ = cKDTree(xy).query(xy, k=2)
nn = d[:, 1]
print("Nearest-neighbour distance between points (metres):")
print(pd.Series(nn).describe().round(0).to_string())
print(f"\nPoints with another point within 250 m: {(nn < 250).sum()} of {len(nn)}")
cells = pd.Series(list(zip((x // 250).astype(int), (y // 250).astype(int))))
print(f"Distinct 250 m cells occupied: {cells.nunique()} (from {len(pos)} points)")