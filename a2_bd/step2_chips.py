"""Step 2: tile paired S2/S1-RTC overlaps into Sen1Floods11-format chips."""
import json, math, os, sys, time, traceback
import numpy as np
import pandas as pd
import geopandas as gpd
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import from_origin, array_bounds
from shapely.geometry import box, shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PIXEL = 8.983152841195215e-05
SIZE = 512
CHIP_SPAN = PIXEL * SIZE
DST_CRS = CRS.from_epsg(4326)

OUT_CHIPS = "chips"
OUT_QC = "qc"
os.makedirs(OUT_CHIPS, exist_ok=True)
os.makedirs(OUT_QC, exist_ok=True)

catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

with open("pairs.json") as f:
    pairs = json.load(f)
print(f"Loaded {len(pairs)} pairs")

PAIR_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
if PAIR_LIMIT:
    pairs = pairs[:PAIR_LIMIT]
    print(f"PILOT: limited to {len(pairs)} pairs")

bd = gpd.read_file("raw/bgd_admin/bgd_admin0.shp")
if bd.crs is None:
    bd = bd.set_crs(4326)
else:
    bd = bd.to_crs(4326)
bd_union = bd.union_all()
print("BD mask loaded, bounds:", bd_union.bounds)

S2_ASSETS = ["B02", "B03", "B04", "B08", "B11", "SCL"]
S1_ASSETS = ["vv", "vh"]

NODATA_SCL = {0, 1, 3, 8, 9, 10}

records = []
window_scan_count = 0
skip_reasons = {}


def get_item(collection, item_id):
    s = catalog.search(collections=[collection], ids=[item_id])
    items = list(s.items())
    if not items:
        return None
    return pc.sign(items[0])


def open_vrt(href, band_index=1, resampling=Resampling.bilinear, dtype="float32", src_nodata=None):
    ds = rasterio.open(href)
    vrt = WarpedVRT(
        ds, crs=DST_CRS, resampling=resampling,
        src_nodata=src_nodata,
    )
    return ds, vrt


def grid_extent(minx, miny, maxx, maxy, pixel, pad_px=8):
    """Snap bounds to the global pixel grid and pad, returning (transform, width, height)."""
    x0 = math.floor(minx / pixel) * pixel - pad_px * pixel
    y1 = math.ceil(maxy / pixel) * pixel + pad_px * pixel
    x1 = math.ceil(maxx / pixel) * pixel + pad_px * pixel
    y0 = math.floor(miny / pixel) * pixel - pad_px * pixel
    width = int(round((x1 - x0) / pixel))
    height = int(round((y1 - y0) / pixel))
    transform = from_origin(x0, y1, pixel, pixel)
    return transform, width, height


def sample_window(vrt_dict, transform, size):
    """Read a size x size window from each opened (grid-aligned) WarpedVRT at given transform."""
    out = {}
    for name, (ds, vrt, resampling, vrt_transform) in vrt_dict.items():
        try:
            win = rasterio.windows.from_bounds(*array_bounds(size, size, transform), transform=vrt_transform)
            data = vrt.read(
                1,
                out_shape=(size, size),
                window=win,
                resampling=resampling,
                out_dtype="float32",
            )
            arr = data.astype("float32")
        except Exception:
            arr = np.zeros((size, size), dtype="float32")
        out[name] = arr
    return out


MAX_PER_TILE = 4
MAX_TOTAL_HIGH = 320
MAX_TOTAL_LOW = 40

kept_high = 0
kept_low = 0
t_start = time.time()

for pair_idx, p in enumerate(pairs):
    if kept_high >= MAX_TOTAL_HIGH and kept_low >= MAX_TOTAL_LOW:
        print("Both caps reached, stopping tile scan.")
        break
    try:
        s2_item = get_item("sentinel-2-l2a", p["s2_id"])
        s1_item = get_item("sentinel-1-rtc", p["s1_id"])
        if s2_item is None or s1_item is None:
            skip_reasons["item_fetch_fail"] = skip_reasons.get("item_fetch_fail", 0) + 1
            continue

        s2_geom = shape(s2_item.geometry)
        s1_geom = shape(s1_item.geometry)
        overlap = s2_geom.intersection(s1_geom).intersection(bd_union)
        if overlap.is_empty:
            skip_reasons["no_overlap_bd"] = skip_reasons.get("no_overlap_bd", 0) + 1
            continue
        minx, miny, maxx, maxy = overlap.bounds

        # Build candidate window origins spread across the overlap on the fixed grid
        n_cols = max(1, int((maxx - minx) / CHIP_SPAN))
        n_rows = max(1, int((maxy - miny) / CHIP_SPAN))
        # cap candidate grid size to keep runtime bounded
        n_cols = min(n_cols, 6)
        n_rows = min(n_rows, 6)

        # open S2 & S1 VRTs once per pair
        s2_hrefs = {a: s2_item.assets[a].href for a in S2_ASSETS if a in s2_item.assets}
        s1_hrefs = {a: s1_item.assets[a].href for a in S1_ASSETS if a in s1_item.assets}
        if len(s2_hrefs) < 6 or len(s1_hrefs) < 2:
            skip_reasons["missing_assets"] = skip_reasons.get("missing_assets", 0) + 1
            continue

        g_transform, g_width, g_height = grid_extent(minx, miny, maxx, maxy, PIXEL)

        vrts = {}
        for a, href in s2_hrefs.items():
            resampling = Resampling.nearest if a == "SCL" else Resampling.bilinear
            ds = rasterio.open(href)
            vrt = WarpedVRT(ds, crs=DST_CRS, resampling=resampling,
                             transform=g_transform, width=g_width, height=g_height)
            vrts[f"s2_{a}"] = (ds, vrt, resampling, g_transform)
        for a, href in s1_hrefs.items():
            ds = rasterio.open(href)
            # nearest, not bilinear: bilinear averages away speckle and Sen1Floods11 (GEE export) keeps it
            vrt = WarpedVRT(ds, crs=DST_CRS, resampling=Resampling.nearest,
                             transform=g_transform, width=g_width, height=g_height)
            vrts[f"s1_{a}"] = (ds, vrt, Resampling.nearest, g_transform)

        s2_dt_str = p["s2_datetime"][:10].replace("-", "")
        tile_kept = 0
        candidates = []
        for r in range(n_rows):
            for c in range(n_cols):
                ox = minx + c * CHIP_SPAN
                oy = maxy - r * CHIP_SPAN
                candidates.append((ox, oy))
        # shuffle deterministically by pair id hash spread, keep spatial spread by stride sampling
        for (ox, oy) in candidates:
            if tile_kept >= MAX_PER_TILE:
                break
            if kept_high >= MAX_TOTAL_HIGH and kept_low >= MAX_TOTAL_LOW:
                break
            window_scan_count += 1
            transform = from_origin(ox, oy, PIXEL, PIXEL)
            cx = ox + CHIP_SPAN / 2
            cy = oy - CHIP_SPAN / 2

            # fraction inside Bangladesh via chip box
            chip_box = box(ox, oy - CHIP_SPAN, ox + CHIP_SPAN, oy)
            inside_frac = chip_box.intersection(bd_union).area / chip_box.area if chip_box.area > 0 else 0
            if inside_frac < 0.90:
                skip_reasons["outside_bd"] = skip_reasons.get("outside_bd", 0) + 1
                continue

            arrs = sample_window(vrts, transform, SIZE)
            b02, b03, b04, b08, b11 = (arrs["s2_B02"], arrs["s2_B03"], arrs["s2_B04"],
                                         arrs["s2_B08"], arrs["s2_B11"])
            scl = arrs["s2_SCL"]
            vv = arrs["s1_vv"]
            vh = arrs["s1_vh"]

            s2_nodata = (b02 <= 0) | np.isnan(b02) | (b03 <= 0) | np.isnan(b03) | \
                        (b04 <= 0) | np.isnan(b04) | (b08 <= 0) | np.isnan(b08) | \
                        (b11 <= 0) | np.isnan(b11)
            scl_int = np.nan_to_num(scl, nan=0).astype(int)
            scl_bad = np.isin(scl_int, list(NODATA_SCL))

            vv_db = 10 * np.log10(np.where(vv > 0, vv, np.nan))
            vh_db = 10 * np.log10(np.where(vh > 0, vh, np.nan))
            s1_nan = np.isnan(vv_db) | np.isnan(vh_db)
            s1_valid_frac = 1.0 - (s1_nan.sum() / s1_nan.size)
            if s1_valid_frac < 0.99:
                skip_reasons["s1_invalid"] = skip_reasons.get("s1_invalid", 0) + 1
                if os.environ.get("DEBUG_S1"):
                    print(f"  DEBUG s1_valid_frac={s1_valid_frac:.3f} vv_nonzero={(vv>0).sum()} vh_nonzero={(vh>0).sum()} vv_max={np.nanmax(vv) if vv.size else None}")
                continue

            nodata_mask = s2_nodata | scl_bad | s1_nan
            nodata_frac = nodata_mask.sum() / nodata_mask.size
            if nodata_frac >= 0.02:
                skip_reasons["nodata_frac"] = skip_reasons.get("nodata_frac", 0) + 1
                continue

            with np.errstate(invalid="ignore", divide="ignore"):
                mndwi = (b03 - b11) / (b03 + b11)
            water = (mndwi > 0) | (scl_int == 6)
            label = np.zeros((SIZE, SIZE), dtype="int16")
            label[water] = 1
            label[nodata_mask] = -1

            valid_n = (label != -1).sum()
            water_frac = (label == 1).sum() / valid_n if valid_n > 0 else 0

            is_high = 0.03 <= water_frac <= 0.85
            is_low = 0.005 <= water_frac < 0.03
            if not (is_high or is_low):
                skip_reasons["water_frac_out_of_range"] = skip_reasons.get("water_frac_out_of_range", 0) + 1
                continue
            if is_high and kept_high >= MAX_TOTAL_HIGH:
                continue
            if is_low and kept_low >= MAX_TOTAL_LOW:
                continue

            chip_id = f"BD_{s2_dt_str}_{p['s2_id']}_{tile_kept}"
            s1_prof = {
                "driver": "GTiff", "dtype": "float32", "count": 2,
                "height": SIZE, "width": SIZE, "crs": DST_CRS,
                "transform": transform, "nodata": np.nan,
            }
            with rasterio.open(f"{OUT_CHIPS}/{chip_id}_S1.tif", "w", **s1_prof) as dst:
                dst.write(vv_db.astype("float32"), 1)
                dst.write(vh_db.astype("float32"), 2)
                dst.set_band_description(1, "VV")
                dst.set_band_description(2, "VH")

            lbl_prof = {
                "driver": "GTiff", "dtype": "int16", "count": 1,
                "height": SIZE, "width": SIZE, "crs": DST_CRS,
                "transform": transform, "nodata": -1,
            }
            with rasterio.open(f"{OUT_CHIPS}/{chip_id}_Label.tif", "w", **lbl_prof) as dst:
                dst.write(label, 1)

            vv_mean_db = float(np.nanmean(vv_db))
            vh_mean_db = float(np.nanmean(vh_db))

            rec = {
                "id": chip_id, "s2_item": p["s2_id"], "s1_item": p["s1_id"],
                "s2_datetime": p["s2_datetime"], "s1_datetime": p["s1_datetime"],
                "gap_hours": p["gap_hours"], "lon_center": cx, "lat_center": cy,
                "water_frac": water_frac, "nodata_frac": nodata_frac,
                "vv_mean_db": vv_mean_db, "vh_mean_db": vh_mean_db,
                "band": "low" if is_low else "high",
                "_s2_arrs": None,
            }
            # stash raw arrays path for QC generation (recompute later from files is simpler;
            # save small npz for QC step to avoid re-downloading)
            np.savez_compressed(
                f"{OUT_QC}/.cache_{chip_id}.npz",
                b02=b02, b03=b03, b04=b04, b08=b08, b11=b11,
                vv_db=vv_db, vh_db=vh_db, label=label,
            )
            del rec["_s2_arrs"]
            records.append(rec)
            tile_kept += 1
            if is_high:
                kept_high += 1
            else:
                kept_low += 1

        for a, (ds, vrt, _, _t) in vrts.items():
            vrt.close()
            ds.close()

        if pair_idx % 5 == 0 or PAIR_LIMIT:
            elapsed = time.time() - t_start
            print(f"[{pair_idx+1}/{len(pairs)}] scanned={window_scan_count} kept_high={kept_high} kept_low={kept_low} elapsed={elapsed:.0f}s", flush=True)

    except Exception as e:
        skip_reasons["exception"] = skip_reasons.get("exception", 0) + 1
        print("ERROR on pair", p.get("s2_id"), repr(e))
        traceback.print_exc()
        continue

print("FINAL kept_high", kept_high, "kept_low", kept_low, "scanned", window_scan_count)
print("skip_reasons", skip_reasons)

df = pd.DataFrame(records)
df.to_csv("index.csv", index=False)
with open("scan_stats.json", "w") as f:
    json.dump({
        "window_scan_count": window_scan_count,
        "kept_high": kept_high,
        "kept_low": kept_low,
        "skip_reasons": skip_reasons,
        "elapsed_sec": time.time() - t_start,
        "n_pairs_used": pair_idx + 1,
    }, f, indent=2)
print("DONE_STEP2")
