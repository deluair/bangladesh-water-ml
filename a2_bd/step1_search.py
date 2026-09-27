"""Step 1: STAC search, S2/S1-RTC pairing, cloud-filter check."""
import json, itertools
from datetime import datetime, timedelta
import pystac_client
import planetary_computer as pc
from shapely.geometry import shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = [88.0, 20.6, 92.7, 26.7]
WINDOWS = ["2024-08-15/2024-09-30", "2024-10-01/2024-10-31"]

catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

# --- Cloud filter check (independent of main pairing) ---
check_bbox = BBOX
check_range = "2024-01-01/2024-03-31"

cql2_filter = {"op": "<", "args": [{"property": "eo:cloud_cover"}, 10]}
s_cql2 = catalog.search(
    collections=["sentinel-2-l2a"], bbox=check_bbox, datetime=check_range,
    filter=cql2_filter, filter_lang="cql2-json",
)
items_cql2 = list(s_cql2.items())
cc_cql2 = [it.properties.get("eo:cloud_cover") for it in items_cql2]

s_query = catalog.search(
    collections=["sentinel-2-l2a"], bbox=check_bbox, datetime=check_range,
    query={"eo:cloud_cover": {"lt": 10}},
)
items_query = list(s_query.items())
cc_query = [it.properties.get("eo:cloud_cover") for it in items_query]

cloud_check = {
    "cql2_count": len(items_cql2),
    "cql2_max_cc": max(cc_cql2) if cc_cql2 else None,
    "query_count": len(items_query),
    "query_max_cc": max(cc_query) if cc_query else None,
}
print("CLOUD_CHECK", json.dumps(cloud_check))
with open("cloud_filter_check.json", "w") as f:
    json.dump(cloud_check, f, indent=2)

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
