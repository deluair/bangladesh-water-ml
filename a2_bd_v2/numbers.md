# Where each v0.2.0 number in the README comes from

All paths are relative to the repository root. "Pooled IoU" = sum of intersections / sum of unions over the chips named.
QC-kept chips = rows of `a2_bd_v2/qc_decisions.csv` with `decision == keep` (323), joined to `a2_bd_v2/index.csv` on `id`.
Check-set chips = QC-kept chips minus the 28 ids in `a2_bd_v2/dedupe_dropped.csv` (295); this is the set `bd_eval.py`
and `bd_finetune.py` score, and what the README calls "kept" everywhere except the Scripts table's description of
`qc_decisions.csv` itself. Tile = the `T..XXX` code in `id`; date = characters 4 to 11 of `id` (the Sentinel-2 date).

## Dedupe (README Part 1, Step 2; technical reference; README_QC.md)

Some Sentinel-2 scenes were reprocessed by ESA to a newer processing baseline after the original chip was built. Chip
ids are `BD_<date>_<platform>_<product>_<sensing time>_<orbit>_<tile>_<processing time>_<chip#>`; splitting on `_`,
field 4 is the Sentinel-2 sensing time, field 6 the tile, field 7 the processing timestamp, field 8 the chip number.
Two rows sharing (field 4, field 6, field 8) but differing in field 7 are the same acquisition, not independent chips.

| README number | Source | Derivation |
|---|---|---|
| 358 reviewed | `qc_decisions.csv` | row count |
| 323 kept by QC | `qc_decisions.csv` `decision` | count of `keep` |
| 295 unique (sensing time, tile, chip#) keys among the 323 | `qc_decisions.csv`, id parsed as above | `a2_bd_v2/step2c_dedupe.py` |
| 28 duplicate groups, 28 dropped | same | every duplicate group found has exactly 2 members, so groups dropped = excess chips dropped = 28 |
| 295 chips in the check set | 323 − 28 | `dedupe_dropped.csv` has 28 rows (id, kept_twin_id); the kept twin is the row with the latest processing timestamp in each group |

`qc_decisions.csv` is not modified by dedupe (it still shows 323 keep / 35 reject, the pre-dedupe QC decision).
`bd_eval.py` and `bd_finetune.py` read `qc_decisions.csv` as before, then additionally drop any id present in
`dedupe_dropped.csv` when that file exists next to it (inert for the v1 check set, which has no such file).

## Check set v2 description (README Part 1, Step 2; notebook 03b; glossary; technical reference)

| README number | Source | Derivation |
|---|---|---|
| 358 chips | `qc_decisions.csv`, `index.csv` | row count of each (358; same ids in both) |
| 323 kept by QC, 35 rejected | `qc_decisions.csv` `decision` | count of `keep` (323) and `reject` (35) |
| 21 / 11 / 3 reject reasons | `qc_decisions.csv` `reason` where `decision == reject` | `unmasked cumulus` 21, `cloud shadow labelled water` 11, `haze/unmasked cumulus` 3 |
| 295 in the check set | 323 QC-kept − 28 deduped (see above) | |
| 27 tiles, 36 dates | check-set chips (295) | distinct tile codes 27; distinct dates 36 (unchanged by dedupe: the dropped 28 share tiles/dates with a kept chip) |
| 88.26°E to 92.27°E, 21.59°N to 26.28°N | `index.csv` `lon_center`, `lat_center`, check-set chips | min/max, 2 decimals (unchanged by dedupe) |
| 4.5 to 19.5 hours | `index.csv` `gap_hours`, check-set chips | min 4.4934, max 19.5393 |
| 292 high, 31 low (QC-kept, 323) | `index.csv` `band`, QC-kept chips | value counts; all 28 duplicates dropped are "high" band, so the check set (295) is 264 high, 31 low |
| 2019 to 2023 | `index.csv` `s2_datetime`; `step1_search.py` `WINDOWS` | check-set chips per year: 2019 98, 2020 21, 2021 26, 2022 105, 2023 45 |
| about 147 training chips | `bd_finetune_results_*.csv` `fold` | fold sizes 148 and 147 in every variant; each fold trains on the other |
| 611 MB, sha256 `7bb45763…39af` | release asset `bd_s1_water_checkset_v2.tar.gz` | 611,004,939 bytes; `shasum -a 256`; rebuilt to add `dedupe_dropped.csv` and the updated `README_QC.md` |

## Zero-shot scores on v2 (README Step 3 table, "0.055 on v2")

`a2_bd_v2/bd_eval_<model>.csv`, 295 rows each (dedupe applied inside `bd_eval.py`); U-Net = `sum(unet_I) / sum(unet_U)`,
Otsu = `sum(otsu_I) / sum(otsu_U)`. The same values are printed in `a2_bd_v2/bd_eval.log`.

| README number | File | Exact value |
|---|---|---|
| 0.273 | `bd_eval_robust_s20260927.csv` | 0.2734 |
| 0.166 | `bd_eval_robust_s1.csv` | 0.1661 |
| 0.108 | `bd_eval_robust_s2.csv` | 0.1081 |
| 0.055 (no radiometric augmentation, `unet_s1_water.pt`) | `bd_eval_l4_model.csv` | 0.0547 |
| Otsu 0.468 | any of the four files (identical Otsu columns) | 0.4681 |

## Fine-tuning on v2 (README Step 4 table, v2 columns)

`a2_bd_v2/bd_finetune_results_<split>_<init>.csv`: one row per (seed, chip), 295 chips per seed (dedupe applied inside
`bd_finetune.py`), no duplicates. Per seed: `sum(I) / sum(U)`; the README gives mean ± sample sd over the three seeds.
Printed in `a2_bd_v2/bd_finetune.log`.

| README row | File | Seed 20260927 / 1 / 2 | Mean ± sd |
|---|---|---|---|
| Sen1Floods11 start, held-out tiles | `bd_finetune_results_tile_s1f11.csv` | 0.8067 / 0.8022 / 0.8057 | 0.8049 ± 0.0024 → 0.805 ± 0.002 |
| Sen1Floods11 start, held-out dates | `bd_finetune_results_date_s1f11.csv` | 0.8061 / 0.8038 / 0.8031 | 0.8043 ± 0.0016 → 0.804 ± 0.002 |
| ImageNet start, held-out tiles | `bd_finetune_results_tile_imagenet.csv` | 0.8058 / 0.7986 / 0.8064 | 0.8036 ± 0.0043 → 0.804 ± 0.004 |
| ImageNet start, held-out dates | `bd_finetune_results_date_imagenet.csv` | 0.7980 / 0.8053 / 0.8068 | 0.8034 ± 0.0047 → 0.803 ± 0.005 |
| Otsu 0.468 (v2 column) | `bd_eval_*.csv` as above; also printed at the end of each run in `bd_finetune.log` | 0.4681 |
| "0.803–0.805" | min and max of the four means above | |
| "at most 0.001" | tiles: 0.8049 − 0.8036 = 0.0013; dates: 0.8043 − 0.8034 = 0.0009 | |

## v0.1.0 numbers kept for comparison

Unchanged from release v0.1.0. v1 zero-shot scores (0.235 / 0.147 / 0.092 / 0.049, Otsu 0.525) were reproduced
exactly for this release by rerunning `a2_bd/bd_eval.py` on the v1 check set; the per-chip files `a2_bd/bd_eval_*.csv`
are byte-identical to the rerun. v1 Apple GPU fine-tuning means and sds come from `a2_bd/bd_finetune_results_*.csv`
(per-seed `sum(I) / sum(U)`: tiles/s1f11 0.8145 ± 0.0037, dates/s1f11 0.8230 ± 0.0095, tiles/imagenet 0.8137 ± 0.0115,
dates/imagenet 0.8088 ± 0.0083). v1 Colab L4 figures are the outputs saved in
`notebooks/03_bangladesh_checkset_finetune.ipynb`. v1 has no reprocessed-duplicate issue (a single 2024 search window,
no dedupe file), so `a2_bd/bd_eval.py` and `a2_bd/bd_finetune.py` run on it exactly as before.
