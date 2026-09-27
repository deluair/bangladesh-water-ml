"""Re-run only rows whose status is error/nodata in a1_results.csv, patch them in place."""
import sys
import pandas as pd
import a1_border_water as a1
out = pd.read_csv("a1_results.csv", index_col=0)
bad = out[out.dry_status.str.startswith("error") | out.monsoon_status.str.startswith("error")]
c = pd.read_csv("data/crossings.csv").loc[bad.index]
border, bd = a1.load_border(sys.argv[1])
res = pd.DataFrame([a1.one(r, border, bd) for r in c.itertuples()]).set_index("idx")
out.update(res)
out.to_csv("a1_results.csv", index=True)
for s in a1.SEASONS:
    print(s, out[f"{s}_status"].value_counts().to_dict())
