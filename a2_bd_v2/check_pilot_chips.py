"""Sanity check pilot chips: label values, S1 band count/dtype, no all-nan/zero S1."""
import glob, sys
import numpy as np
import rasterio

chips_dir = sys.argv[1] if len(sys.argv) > 1 else "chips"
s1_files = sorted(glob.glob(f"{chips_dir}/*_S1.tif"))
print(f"Found {len(s1_files)} S1 chips in {chips_dir}")

bad = []
for s1_path in s1_files:
    base = s1_path[:-len("_S1.tif")]
    lbl_path = f"{base}_Label.tif"
    with rasterio.open(s1_path) as ds:
        arr = ds.read()
        n_bands = ds.count
        dtype = ds.dtypes[0]
    vv, vh = arr[0], arr[1]
    vv_finite = np.isfinite(vv)
    vh_finite = np.isfinite(vh)
    vv_allbad = (~vv_finite).all() or (vv_finite & (vv == 0)).sum() == vv_finite.sum() and vv_finite.sum() > 0
    vh_allbad = (~vh_finite).all()
    vv_zero_frac = (vv[vv_finite] == 0).mean() if vv_finite.any() else 1.0
    vh_zero_frac = (vh[vh_finite] == 0).mean() if vh_finite.any() else 1.0

    with rasterio.open(lbl_path) as ds:
        label = ds.read(1)
    uniq = sorted(set(np.unique(label).tolist()))
    label_ok = all(v in (-1, 0, 1) for v in uniq)

    issues = []
    if n_bands != 2:
        issues.append(f"n_bands={n_bands}")
    if dtype != "float32":
        issues.append(f"dtype={dtype}")
    if not vv_finite.any():
        issues.append("vv_all_nonfinite")
    if not vh_finite.any():
        issues.append("vh_all_nonfinite")
    if vv_zero_frac > 0.99:
        issues.append(f"vv_zero_frac={vv_zero_frac:.3f}")
    if vh_zero_frac > 0.99:
        issues.append(f"vh_zero_frac={vh_zero_frac:.3f}")
    if not label_ok:
        issues.append(f"label_values={uniq}")

    if issues:
        bad.append((base, issues))
    else:
        print(f"OK {base}: label_values={uniq} vv_zero_frac={vv_zero_frac:.4f} vh_zero_frac={vh_zero_frac:.4f}")

print()
if bad:
    print(f"BAD chips: {len(bad)}")
    for b, issues in bad:
        print(" ", b, issues)
else:
    print("All chips passed sanity checks.")
