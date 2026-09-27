"""Rebuild index.csv from whatever chips/*.tif + qc/.cache_*.npz exist, robust to an
early-terminated step2 run. Pulls s1_item/s2_datetime/s1_datetime/gap_hours from pairs.json
by matching the embedded s2 item id in the chip filename, and recomputes water_frac,
nodata_frac, vv/vh mean dB, lon/lat center directly from the written GeoTIFFs.
"""
import glob, json, os, re
import numpy as np
import pandas as pd
import rasterio

with open("pairs.json") as f:
    pairs = json.load(f)
pairs_by_s2 = {p["s2_id"]: p for p in pairs}

s1_files = sorted(glob.glob("chips/*_S1.tif"))
print(f"Found {len(s1_files)} S1 chip files")

records = []
for s1_path in s1_files:
    base = os.path.basename(s1_path)[:-len("_S1.tif")]
    lbl_path = f"chips/{base}_Label.tif"
    if not os.path.exists(lbl_path):
        print("missing label for", base)
        continue
    # chip id format: BD_<YYYYMMDD>_<s2_item_id>_<n>
    m = re.match(r"BD_(\d{8})_(.+)_(\d+)$", base)
    if not m:
        print("id parse fail", base)
        continue
    ymd, s2_id, n = m.group(1), m.group(2), m.group(3)
    p = pairs_by_s2.get(s2_id)
    if p is None:
        print("no pair match for", s2_id)
        continue

    with rasterio.open(s1_path) as ds:
        vv = ds.read(1)
        vh = ds.read(2)
        b = ds.bounds
        cx = (b.left + b.right) / 2
        cy = (b.top + b.bottom) / 2
    with rasterio.open(lbl_path) as ds:
        label = ds.read(1)

    valid_n = (label != -1).sum()
    water_frac = float((label == 1).sum() / valid_n) if valid_n > 0 else float("nan")
    nodata_frac = float((label == -1).sum() / label.size)
    vv_mean_db = float(np.nanmean(vv))
    vh_mean_db = float(np.nanmean(vh))
    band = "low" if 0.005 <= water_frac < 0.03 else "high"

    records.append({
        "id": base, "s2_item": p["s2_id"], "s1_item": p["s1_id"],
        "s2_datetime": p["s2_datetime"], "s1_datetime": p["s1_datetime"],
        "gap_hours": p["gap_hours"], "lon_center": cx, "lat_center": cy,
        "water_frac": water_frac, "nodata_frac": nodata_frac,
        "vv_mean_db": vv_mean_db, "vh_mean_db": vh_mean_db, "band": band,
    })

df = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
df.to_csv("index.csv", index=False)
print(f"Wrote index.csv with {len(df)} rows")
print(df["band"].value_counts())
print("gap_hours stats:", df["gap_hours"].describe())
print("vv_mean_db mean:", df["vv_mean_db"].mean(), "vh_mean_db mean:", df["vh_mean_db"].mean())
print("DONE_REINDEX")
