# Where each v0.2.0 number in the README comes from

All paths are relative to the repository root. "Pooled IoU" = sum of intersections / sum of unions over the chips named.
Kept chips = rows of `a2_bd_v2/qc_decisions.csv` with `decision == keep`, joined to `a2_bd_v2/index.csv` on `id`.
Tile = the `T..XXX` code in `id`; date = characters 4 to 11 of `id` (the Sentinel-2 date).

## Check set v2 description (README Part 1, Step 2; notebook 03b; glossary; technical reference)

| README number | Source | Derivation |
|---|---|---|
| 358 chips | `qc_decisions.csv`, `index.csv` | row count of each (358; same ids in both) |
| 323 kept, 35 rejected | `qc_decisions.csv` `decision` | count of `keep` (323) and `reject` (35) |
| 21 / 11 / 3 reject reasons | `qc_decisions.csv` `reason` where `decision == reject` | `unmasked cumulus` 21, `cloud shadow labelled water` 11, `haze/unmasked cumulus` 3 |
| 27 tiles, 36 dates | kept chips | distinct tile codes 27; distinct dates 36 |
| 88.26°E to 92.27°E, 21.59°N to 26.28°N | `index.csv` `lon_center`, `lat_center`, kept chips | min/max, 2 decimals |
| 4.5 to 19.5 hours | `index.csv` `gap_hours` | min 4.48 (all chips), 4.49 (kept); max 19.54 |
| 292 high, 31 low | `index.csv` `band`, kept chips | value counts |
| 2019 to 2023 | `index.csv` `s2_datetime`; `step1_search.py` `WINDOWS` | kept per year: 2019 98, 2020 21, 2021 26, 2022 109, 2023 69 |
| about 161 training chips | `bd_finetune_results_*.csv` `fold` | fold sizes 162 and 161 in every variant; each fold trains on the other |
| 611 MB, sha256 `01482b94…49de` | release asset `bd_s1_water_checkset_v2.tar.gz` | 611,007,382 bytes; `shasum -a 256` |

## Zero-shot scores on v2 (README Step 3 table, "0.056 on v2")

`a2_bd_v2/bd_eval_<model>.csv`, 323 rows each; U-Net = `sum(unet_I) / sum(unet_U)`, Otsu = `sum(otsu_I) / sum(otsu_U)`.
The same values are printed in `a2_bd_v2/bd_eval.log`.

| README number | File | Exact value |
|---|---|---|
| 0.266 | `bd_eval_robust_s20260927.csv` | 0.2659 |
| 0.161 | `bd_eval_robust_s1.csv` | 0.1611 |
| 0.106 | `bd_eval_robust_s2.csv` | 0.1058 |
| 0.056 (no radiometric augmentation, `unet_s1_water.pt`) | `bd_eval_l4_model.csv` | 0.0559 |
| Otsu 0.472 | any of the four files (identical Otsu columns) | 0.4719 |

## Fine-tuning on v2 (README Step 4 table, v2 columns)

`a2_bd_v2/bd_finetune_results_<split>_<init>.csv`: one row per (seed, chip), 323 chips per seed, no duplicates.
Per seed: `sum(I) / sum(U)`; the README gives mean ± sample sd over the three seeds. Printed in `a2_bd_v2/bd_finetune.log`.

| README row | File | Seed 20260927 / 1 / 2 | Mean ± sd |
|---|---|---|---|
| Sen1Floods11 start, held-out tiles | `bd_finetune_results_tile_s1f11.csv` | 0.7929 / 0.8063 / 0.8061 | 0.8018 ± 0.0077 → 0.802 ± 0.008 |
| Sen1Floods11 start, held-out dates | `bd_finetune_results_date_s1f11.csv` | 0.8032 / 0.8037 / 0.8086 | 0.8052 ± 0.0030 → 0.805 ± 0.003 |
| ImageNet start, held-out tiles | `bd_finetune_results_tile_imagenet.csv` | 0.8030 / 0.7960 / 0.7974 | 0.7988 ± 0.0037 → 0.799 ± 0.004 |
| ImageNet start, held-out dates | `bd_finetune_results_date_imagenet.csv` | 0.8094 / 0.8049 / 0.8076 | 0.8073 ± 0.0023 → 0.807 ± 0.002 |
| Otsu 0.472 (v2 column) | `bd_eval_*.csv` as above; also printed at the end of each run in `bd_finetune.log` | 0.4719 |
| "0.799–0.807" | min and max of the four means above | |
| "at most 0.003" | tiles: 0.8018 − 0.7988 = 0.0030; dates: 0.8073 − 0.8052 = 0.0021 | |

## v0.1.0 numbers kept for comparison

Unchanged from release v0.1.0. v1 zero-shot scores (0.235 / 0.147 / 0.092 / 0.049, Otsu 0.525) were reproduced
exactly for this release by rerunning `a2_bd/bd_eval.py` on the v1 check set; the per-chip files `a2_bd/bd_eval_*.csv`
are byte-identical to the rerun. v1 Apple GPU fine-tuning means and sds come from `a2_bd/bd_finetune_results_*.csv`
(per-seed `sum(I) / sum(U)`: tiles/s1f11 0.8145 ± 0.0037, dates/s1f11 0.8230 ± 0.0095, tiles/imagenet 0.8137 ± 0.0115,
dates/imagenet 0.8088 ± 0.0083). v1 Colab L4 figures are the outputs saved in
`notebooks/03_bangladesh_checkset_finetune.ipynb`.
