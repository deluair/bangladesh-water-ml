import pandas as pd
d = pd.read_csv("a1_results.csv")
print("errors dry/monsoon:", d.dry_status.str.startswith("error").sum(), d.monsoon_status.str.startswith("error").sum())
for s in ("dry","monsoon"):
    d[s+"_c"] = d[s+"_status"].eq("confirmed")
d["any_c"] = d.dry_c | d.monsoon_c
g = d.groupby("RIVER_ID").agg(dry=("dry_c","any"), mon=("monsoon_c","any"), anyc=("any_c","any"), occ=("gsw_max_occ","max"), up=("UPLAND_SKM","max"))
print("rivers:", len(g), "dry-confirmed:", int(g.dry.sum()), "monsoon-confirmed:", int(g.mon.sum()), "either:", int(g.anyc.sum()))
g["cls"] = pd.cut(g.occ, [-1,9.999,49.999,101], labels=["<10","10-49",">=50"])
print(g.groupby("cls", observed=True)[["dry","mon","anyc"]].agg(["sum","count"]))
print("crossings confirmed either season:", int(d.any_c.sum()))
print("median width m dry/mon:", d.loc[d.dry_c,"dry_width_m"].median(), d.loc[d.monsoon_c,"monsoon_width_m"].median())
print(d.groupby("division")[["dry_c","monsoon_c"]].sum())
