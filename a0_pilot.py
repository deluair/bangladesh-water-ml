"""A0 pilot: Sentinel-2 (10 m) water check at border river crossings, dry vs monsoon season.

For each crossing: search Microsoft Planetary Computer (sentinel-2-l2a, no login) for the
least-cloudy scene in a dry window (Jan-Mar) and a monsoon window (Aug-Oct, cloud filter looser),
read a 300 m chip of B03 (green) and B08 (NIR) at 10 m, compute NDWI = (G - NIR)/(G + NIR),
and report the water share within 60 m of the crossing (NDWI > 0) next to Landsat's JRC occurrence.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

YEAR = 2024
WINDOWS = {"dry": (f"{YEAR}-01-01/{YEAR}-03-31", 10), "monsoon": (f"{YEAR}-08-01/{YEAR}-10-31", 40)}
HALF = 150  # metres: chip half-width
R_WATER = 60  # metres: radius for water share

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)


def chip(item, band: str, lon: float, lat: float) -> tuple[np.ndarray, float]:
    href = item.assets[band].href
    with rasterio.open(href) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        win = from_bounds(x[0] - HALF, y[0] - HALF, x[0] + HALF, y[0] + HALF, src.transform)
        return src.read(1, window=win, boundless=True, fill_value=0).astype("float32"), src.res[0]


def water_share(lon: float, lat: float, window: str, cloud: int) -> dict:
    items = list(cat.search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                            datetime=window, query={"eo:cloud_cover": {"lt": cloud}}).items())
    if not items:
        return {"scene": None, "water_share": None}
    it = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    g, res = chip(it, "B03", lon, lat)
    n, _ = chip(it, "B08", lon, lat)
    ndwi = (g - n) / np.maximum(g + n, 1)
    h, w = ndwi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) * res**2 <= R_WATER**2
    return {"scene": it.id, "cloud": round(it.properties["eo:cloud_cover"], 1),
            "water_share": round(float((ndwi[mask] > 0).mean()), 3)}


def main(n: int = 20) -> None:
    c = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/crossings.csv")
    # stratified pilot: persistent, occasional, and not-seen-by-Landsat crossings
    rng = np.random.default_rng(20260927)
    parts = [c[c.gsw_max_occ >= 50], c[(c.gsw_max_occ >= 10) & (c.gsw_max_occ < 50)], c[c.gsw_max_occ < 10]]
    sample = pd.concat([p.iloc[rng.choice(len(p), n // 3 + (i == 2) * (n % 3), replace=False)] for i, p in enumerate(parts)])
    rows = []
    for _, r in sample.iterrows():
        row = {"RIVER_ID": r.RIVER_ID, "lat": r.lat, "lon": r.lon, "gsw_max_occ": r.gsw_max_occ, "upland_km2": r.UPLAND_SKM}
        for k, (win, cloud) in WINDOWS.items():
            res = water_share(r.lon, r.lat, win, cloud)
            row[f"{k}_water"] = res["water_share"]
            row[f"{k}_cloud"] = res.get("cloud")
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("a0_pilot_results.csv", index=False)
    print(out.groupby(pd.cut(out.gsw_max_occ, [-1, 9.99, 49.99, 100], labels=["<10", "10-49", ">=50"]),
                      observed=True)[["dry_water", "monsoon_water"]].mean().round(3))


if __name__ == "__main__":
    main()
