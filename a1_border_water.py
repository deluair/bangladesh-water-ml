"""A1: does a connected water body actually cross the Bangladesh-India border near each modelled crossing?

Per crossing and season (dry Jan-Mar, monsoon Aug-Oct; YEAR): least-cloudy Sentinel-2 L2A scene from
Microsoft Planetary Computer (CQL2 cloud filter), 1 km chip of B03/B08 at 10 m, NDWI > 0 water mask,
morphological clean-up, connected components. The India-facing border (BBS adminlines adm_level 0,
nearer India than Myanmar) is rasterised into the chip. A crossing is CONFIRMED in a season when a
water component touches the border within SEARCH_M of the modelled point and has water pixels on both
sides of the line. Reports the border-crossing width (water pixels on the border line x 10 m).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import Point, box
from shapely.ops import unary_union

YEAR = 2024
SEASONS = {"dry": (f"{YEAR}-01-01/{YEAR}-03-31", 10), "monsoon": (f"{YEAR}-08-01/{YEAR}-10-31", 40)}
HALF = 500  # metres
SEARCH_M = 500

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def load_border(raw: str) -> gpd.GeoSeries:
    lines = gpd.read_file(f"{raw}/bbs/bgd_adminlines.shp").to_crs(32646)
    intl = lines[lines.adm_level == 0].copy()
    ind = unary_union(gpd.read_file(f"{raw}/ind_adm0.geojson").to_crs(32646).geometry)
    mmr = unary_union(gpd.read_file(f"{raw}/mmr_adm0.geojson").to_crs(32646).geometry)
    keep = intl.geometry.distance(ind) <= intl.geometry.distance(mmr)
    bd = unary_union(gpd.read_file(f"{raw}/bbs/bgd_admin0.shp").to_crs(32646).geometry)
    return gpd.GeoSeries([unary_union(intl[keep].geometry)], crs=32646), gpd.GeoSeries([bd], crs=32646)


def scene(lon: float, lat: float, window: str, cloud: int):
    items = list(cat.search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                            datetime=window, filter_lang="cql2-json",
                            filter={"op": "<", "args": [{"property": "eo:cloud_cover"}, cloud]}).items())
    return min(items, key=lambda i: i.properties["eo:cloud_cover"]) if items else None


def one(r, border, bd) -> dict:
    out = {"idx": r.Index}
    pt = gpd.GeoSeries([Point(r.lon, r.lat)], crs=4326)
    for season, (win, cloud) in SEASONS.items():
        try:
            it = scene(r.lon, r.lat, win, cloud)
            if it is None:
                out[f"{season}_status"] = "no_scene"
                continue
            with rasterio.open(it.assets["B03"].href) as g_src, rasterio.open(it.assets["B08"].href) as n_src:
                p = pt.to_crs(g_src.crs).iloc[0]
                win_ = from_bounds(p.x - HALF, p.y - HALF, p.x + HALF, p.y + HALF, g_src.transform)
                g = g_src.read(1, window=win_, boundless=True, fill_value=0).astype("float32")
                n = n_src.read(1, window=win_, boundless=True, fill_value=0).astype("float32")
                tr = g_src.window_transform(win_)
                crs = g_src.crs
            if (g == 0).mean() > 0.2:
                out[f"{season}_status"] = "nodata"
                continue
            water = ((g - n) / np.maximum(g + n, 1)) > 0
            water = ndimage.binary_opening(water, iterations=1)
            lab, _ = ndimage.label(water)
            chipgeom = box(p.x - HALF, p.y - HALF, p.x + HALF, p.y + HALF)
            b = border.to_crs(crs).iloc[0].intersection(chipgeom)
            if b.is_empty:
                out[f"{season}_status"] = "border_outside_chip"
                continue
            near = b.intersection(p.buffer(SEARCH_M))
            bline = rasterize([(near.buffer(10), 1)], out_shape=g.shape, transform=tr, fill=0).astype(bool)
            inside = rasterize([(bd.to_crs(crs).iloc[0], 1)], out_shape=g.shape, transform=tr, fill=0).astype(bool)
            best_w, confirmed = 0, False
            for k in np.unique(lab[bline & water]):
                comp = lab == k
                both = (comp & inside & ~bline).any() and (comp & ~inside & ~bline).any()
                width = int((comp & bline).sum() / 2)  # buffered line is ~2 px thick
                if both:
                    confirmed = True
                    best_w = max(best_w, width * 10)
            out[f"{season}_status"] = "confirmed" if confirmed else "no_water_crossing"
            out[f"{season}_width_m"] = best_w
            out[f"{season}_scene"] = it.id
            out[f"{season}_cloud"] = round(it.properties["eo:cloud_cover"], 1)
        except Exception as exc:  # keep going; record why
            out[f"{season}_status"] = f"error:{type(exc).__name__}"
    return out


def main() -> None:
    csv, raw = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    c = pd.read_csv(csv)
    if limit:
        c = c.sample(limit, random_state=20260927)
    border, bd = load_border(raw)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: one(r, border, bd), c.itertuples()))
    r = pd.DataFrame(res).set_index("idx")
    out = c.join(r)
    out.to_csv("a1_results.csv", index=True)
    for s in SEASONS:
        print(s, out[f"{s}_status"].value_counts().to_dict())
    grp = pd.cut(out.gsw_max_occ, [-1, 9.99, 49.99, 100], labels=["<10", "10-49", ">=50"])
    print((out.assign(dry=out.dry_status.eq("confirmed"), monsoon=out.monsoon_status.eq("confirmed"))
           .groupby(grp, observed=True)[["dry", "monsoon"]].mean().round(3)))


if __name__ == "__main__":
    main()
