"""Step 1: STAC search, S2/S1-RTC pairing, cloud-filter check."""
import json, itertools
from datetime import datetime, timedelta
import pystac_client
import planetary_computer as pc
from shapely.geometry import shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [88.0, 20.6, 92.7, 26.7]
WINDOWS = [
    "2019-07-01/2019-08-31", "2019-09-01/2019-10-31",
    "2020-07-01/2020-08-31", "2020-09-01/2020-10-31",
    "2021-07-01/2021-08-31", "2021-09-01/2021-10-31",
    "2022-07-01/2022-08-31", "2022-09-01/2022-10-31",
    "2023-07-01/2023-08-31", "2023-09-01/2023-10-31",
]

catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

# --- Main S2 search across both windows ---
s2_items = []
for w in WINDOWS:
    s = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=w,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    its = list(s.items())
    print(f"S2 window {w}: {len(its)} items")
    s2_items.extend(its)

print("Total S2 candidates:", len(s2_items))

# --- Pair each S2 item with S1 RTC item within 24h, footprint intersecting ---
pairs = []
for s2 in s2_items:
    s2_dt = datetime.fromisoformat(s2.properties["datetime"].replace("Z", "+00:00"))
    s2_geom = shape(s2.geometry)
    t0 = (s2_dt - timedelta(hours=24)).isoformat()
    t1 = (s2_dt + timedelta(hours=24)).isoformat()
    try:
        s1_search = catalog.search(
            collections=["sentinel-1-rtc"],
            intersects=s2.geometry,
            datetime=f"{t0}/{t1}",
        )
        s1_items = list(s1_search.items())
    except Exception as e:
        print("S1 search error for", s2.id, e)
        continue
    best = None
    best_gap = None
    for s1 in s1_items:
        s1_dt = datetime.fromisoformat(s1.properties["datetime"].replace("Z", "+00:00"))
        s1_geom = shape(s1.geometry)
        if not s1_geom.intersects(s2_geom):
            continue
        gap = abs((s1_dt - s2_dt).total_seconds()) / 3600.0
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = s1
    if best is not None:
        pairs.append({
            "s2_id": s2.id,
            "s2_datetime": s2.properties["datetime"],
            "s2_cloud": s2.properties.get("eo:cloud_cover"),
            "s1_id": best.id,
            "s1_datetime": best.properties["datetime"],
            "gap_hours": best_gap,
        })

print("Total S2/S1-RTC pairs found:", len(pairs))
with open("pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

# Save S2 item ids list by window for reference
with open("s2_search_counts.json", "w") as f:
    json.dump({w: None for w in WINDOWS}, f)
print("DONE_STEP1")
