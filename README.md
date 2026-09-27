# bangladesh-water-ml

Free satellite data and machine learning applied to two questions about Bangladesh's water:

1. **Border rivers.** River maps show hundreds of channels crossing the Bangladesh–India border. At which of these
   modelled crossings can a satellite actually see water flowing across the line, in the dry season and in the monsoon?
2. **Flood mapping from radar.** Clouds hide Bangladesh's floods from optical satellites for most of the monsoon. Radar
   sees through cloud. Does a radar flood-mapping model trained on floods elsewhere in the world work over Bangladesh,
   and if not, what does it take to make it work?

Everything here uses public data, runs on a laptop or a free/paid Google Colab GPU, and ships with the numbers it
produced. Counts below were re-derived from the files in this repository on 2026-09-27.

---

## Part 1 · In plain language

### What the satellites see

* **Sentinel-2** (European Space Agency, free) photographs the ground in visible and infrared light every 5 days at 10 m
  detail. Water absorbs infrared light, so water pixels are easy to separate from land with a simple index (NDWI or
  MNDWI, see the glossary). Its weakness: it cannot see through cloud, and Bangladesh is cloudy for most of June to
  October, exactly when floods happen.
* **Sentinel-1** (also free) is a radar. It sends microwave pulses and records how much bounces back. Calm water acts
  like a mirror and sends the pulse away from the satellite, so water looks dark; land, crops and buildings scatter the
  pulse back and look bright. Radar works through cloud and at night. Its weakness: the image is grainy ("speckle"),
  and wind on water, wet soil, or flooded vegetation can confuse it.

Both are read from **Microsoft Planetary Computer**, which hosts the imagery for free with no login.

### Question 1: which border rivers really cross?

A river network model (HydroRIVERS) laid over the official Bangladesh boundary produces **1,675 points** where a modelled
channel with a catchment of at least 10 km² crosses the India-facing border, belonging to **624 distinct rivers**
(961 flowing into Bangladesh, 714 flowing out). A model line on a map is not proof that water crosses there: many are
small seasonal streams, some are artefacts of the model.

For each crossing the pipeline takes the clearest Sentinel-2 image of a dry-season window (January to March 2024, under
10% cloud) and of a monsoon window (August to October 2024, under 40% cloud), cuts a 1 km square around the point, marks
water pixels, and asks one strict question: **is there one continuous body of water that touches the border within
500 m of the point and has water on both sides of the line?** If yes, the crossing is "confirmed" for that season, and
the width of water along the border line is measured.

**What it found**

| | Dry season | Monsoon | Either season |
|---|---|---|---|
| Crossings confirmed, of 1,675 | 257 | 371 | 405 |
| Rivers with at least one confirmed crossing, of 624 | 105 | 147 | 157 |
| Median width of water on the border line, confirmed crossings | 160 m | 220 m | |

The result lines up with an independent record: the JRC Global Surface Water map, which counts how often Landsat has
seen water at each spot since 1984. Of the 87 rivers with at least one crossing point that was water at least half the
time, 80 are confirmed here; of the 434 rivers whose crossing points were all water less than 10% of the time, 21 are. Confirmation is concentrated
in Rangpur (82 dry / 118 monsoon crossings) and Sylhet (77 / 96), the divisions where the big Himalayan and Meghalaya
rivers enter. The unconfirmed majority is consistent with small channels narrower than a 10 m pixel or dry at the image
date; it does not show that those rivers do not exist.

### Question 2: a radar flood model for Bangladesh

**Step 1, train on the world.** The standard open training set for radar flood mapping is **Sen1Floods11**: 11 flood
events around the world, with 446 chips (512 × 512 pixel image squares) hand-labelled by analysts and 4,384 chips labelled
automatically from optical images. It contains no Bangladesh chips. A U-Net (a standard image-segmentation network, see
glossary) was trained on it on a Colab A100 GPU, three times with different random seeds, to show how stable the result
is.

**Step 2, build a Bangladesh test set.** To test the model on Bangladesh, the project built its own labelled chips from
the 2024 flood season (August to October): each chip pairs a Sentinel-1 radar image with a cloud-free part of a
Sentinel-2 optical image taken 4.6 to 19.5 hours apart. The optical image supplies the answer key (water or not, per
pixel); clouds and cloud shadows are masked out and ignored. Every one of the **78 chips** was inspected by eye.
**17 were rejected**, all for optical defects that make the answer key wrong (13 with cumulus clouds the cloud mask
missed, 4 with haze and cumulus). No chip was rejected because the radar model disagreed with it: rejecting chips the
model gets wrong would rig the test. **61 chips remain**, from 19 Sentinel-2 tiles and 8 dates, spread from 88.95°E to
92.26°E and 21.59°N to 25.26°N.

**Step 3, score it.** The score is IoU (intersection over union): the overlap between predicted water and true water,
divided by their combined area. 1.0 is perfect; 0 is no overlap. It is pooled over all chips. The yardstick is the
**Otsu threshold**, the textbook non-ML method: in each chip, pick the brightness cut-off that best splits dark from
bright pixels and call the dark side water.

| Test set | U-Net, 3 seeds | Otsu |
|---|---|---|
| Sen1Floods11 test split, 90 chips | 0.661 / 0.665 / 0.666 | 0.210 |
| Bolivia (a country held out of training), 15 chips | 0.648 / 0.698 / 0.697 | 0.351 |
| **Bangladesh, 61 chips, model used as trained** | **0.235 / 0.147 / 0.092** | **0.525** |

The model that beats Otsu by three times on the world's test set does far worse than Otsu over Bangladesh.

**Why.** Two causes were measured, not guessed.

1. *The radar numbers are on a different scale.* Sen1Floods11 was built from one processing of Sentinel-1 (called
   sigma0, from Google Earth Engine). Planetary Computer serves a terrain-corrected version (gamma0 RTC). Over Bangladesh
   land it reads about 3.5 dB brighter in the VV channel and 5 dB brighter in VH. A model that learned "water is darker
   than −15 dB" is then simply miscalibrated. Otsu is immune because it picks a fresh cut-off in every chip.
2. *The model relies on speckle.* Smoothing the grain out of a chip the model handles well (from India, IoU 0.735) drops
   it to 0.03. Reprojecting radar with the common "bilinear" method smooths it in exactly this way, so the Bangladesh
   chips here use "nearest neighbour" reprojection, which keeps the grain.

Training with deliberately shifted brightness and occasional smoothing (radiometric augmentation) keeps the world
scores intact but lifts Bangladesh only a little: the three seeds above are that robust version; an earlier run without
augmentation scores 0.049.

**Step 4, teach it Bangladesh.** The fix is to show the model Bangladesh. The 61 chips are split in two groups; the
model is fine-tuned on one group and scored on the other, then the roles swap, so **every chip is scored by a model that
never saw it**. The groups are formed by Sentinel-2 tile (so the test chips come from places the model never saw) or,
separately, by date (so they come from days it never saw). Each variant ran with three seeds, twice: once on an Apple
M-series GPU (`a2_bd/bd_finetune.py`) and once on a Colab L4 GPU (notebook 03, as committed). GPU arithmetic is not
bit-identical across hardware, so the two runs differ slightly.

| Fine-tuning variant (about 30 training chips each time) | IoU, Apple GPU (mean ± sd, 3 seeds) | IoU, Colab L4 | Otsu |
|---|---|---|---|
| Starting from the Sen1Floods11 model, held-out tiles | **0.815 ± 0.004** | **0.830 ± 0.004** | 0.525 |
| Starting from the Sen1Floods11 model, held-out dates | **0.823 ± 0.009** | **0.823 ± 0.009** | 0.525 |
| Starting from a generic ImageNet network, held-out tiles | 0.814 ± 0.011 | 0.820 ± 0.004 | 0.525 |
| Starting from a generic ImageNet network, held-out dates | 0.809 ± 0.008 | 0.801 ± 0.021 | 0.525 |

About 30 local chips take the model from far below Otsu (0.525) to 0.80–0.83. Whether it starts from the global flood
model or from a generic image network barely matters (at most 0.022 IoU in either run): **the local labels do the work,
the global training set adds little here.** That is the practical lesson for anyone mapping floods in Bangladesh with open radar.

### Where AI is and is not used

Machine learning is used only for the flood model (the U-Net). The border-river analysis is plain image arithmetic with
fixed rules. No language model wrote any number, label or result in this repository; every figure above is printed by
the code and files listed below. The Bangladesh answer key comes from optical satellite measurements, checked chip by
chip by a person.

### What can go wrong, and the safety nets

* **Cloud the mask misses** makes a wrong answer key. Net: every chip inspected; the 17 bad ones rejected with a written
  reason in `a2_bd/qc_decisions.csv`.
* **Radar and optical images a day apart** can disagree if water rose or fell in between. Net: pairs are at most 19.5 h
  apart; the effect remains a source of noise in both directions.
* **A small test set** (61 chips) gives wide uncertainty. Net: three seeds, two different ways of holding data out, and
  a fixed baseline on the same chips. The results are a strong signal, not a precise national accuracy figure.
* **Tuning on the test set** would inflate scores. Net: the Bangladesh chips were never used to choose settings for the
  global model; fine-tuning scores come only from chips held out of that run.
* **Border-river misses** can come from cloud, image date or narrow channels. Net: two seasons, a 500 m search radius,
  and an independent comparison with 40 years of Landsat water history.

---

## Part 2 · Technical reference (verified 2026-09-27)

### Notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| `notebooks/01_border_rivers_s2.ipynb` | A1 on a sample (`N = 40`, set `N = 0` for all 1,675); downloads boundaries and `data/crossings.csv` | CPU, minutes |
| `notebooks/01_border_rivers_s2_colab_executed.ipynb` | The same, as executed on Colab (40-crossing sample) | |
| `notebooks/02_flood_radar_unet.ipynb` | Downloads Sen1Floods11 v1.1 from `gs://sen1floods11`, trains the 3-seed robust U-Net, scores test and Bolivia | GPU; A100 about 50 min |
| `notebooks/03_bangladesh_checkset_finetune.ipynb` | Downloads the check set and weights from release v0.1.0 (SHA-256 verified), scores zero-shot, runs all fine-tuning variants | GPU, about 10 min |

Notebooks 02 and 03 are generated by `notebooks/a2/build_nb.py` and `build_nb03.py` (run from the repository root).

### Scripts and outputs

| Path | Role |
|---|---|
| `data/crossings.csv` | 1,675 crossings: HydroRIVERS ids, direction, catchment (km²), mean discharge (m³/s), Strahler order, JRC GSW max occurrence (%), division, lon/lat |
| `a0_pilot.py` | Early pilot of the per-crossing method |
| `a1_border_water.py` | A1 for every crossing (`python a1_border_water.py data/crossings.csv <raw_dir>`; `raw_dir` holds the BBS and geoBoundaries files notebook 01 downloads) |
| `a1_retry.py` | Re-runs rows that failed; `a1_results.csv` has 0 errors in both seasons |
| `a1_summary.py` | Prints every A1 figure in Part 1 |
| `a2_bd/step1_search.py` | Planetary Computer search: S2 L2A (< 20% cloud) paired with the nearest S1 RTC scene within 24 h |
| `a2_bd/step2_chips.py` | 512 × 512 chips at 8.98e-5° (about 10 m): S1 VV/VH in dB (nearest-neighbour warp); label 1 if MNDWI(B03, B11) > 0 or SCL = 6, 0 otherwise, −1 where SCL ∈ {0, 1, 3, 8, 9, 10}; skip if ≥ 2% no-data; "high" band 3–85% water, "low" band 0.5–3%; at most 4 chips per scene pair |
| `a2_bd/step3_qc.py` | Contact sheets for visual QC |
| `a2_bd/index.csv`, `qc_decisions.csv` | 78 chips with scene ids, times, gap, location, water share; keep/reject with reason (61 keep: 50 high, 11 low) |
| `a2_bd/bd_eval.py` | Zero-shot score of one model on the kept chips: `python bd_eval.py qc_decisions.csv <weights.pt> <out.csv>` (run inside the extracted check set) |
| `a2_bd/bd_finetune.py` | Fine-tuning comparison; `SPLIT=tile|date`, `INIT=s1f11|imagenet`, `WEIGHTS=<dir>`; 2-fold greedy balance, 25 epochs, AdamW 1e-4, batch 4 |
| `a2_bd/bd_finetune_results_*.csv`, `bd_finetune.log`, `bd_ablation.log` | The four fine-tuning variants (per chip, per seed) and their printed summaries |
| `a2_bd/bd_eval_*.csv` | Per-chip zero-shot scores |
| `models/a2_results_s1f11.csv`, `a2_results_robust_s1f11.csv` | Sen1Floods11 test and Bolivia scores of the Colab runs |

### Model

`segmentation_models_pytorch.Unet('resnet34', in_channels=2, classes=1)`. Input: VV and VH in dB, clipped to [−50, 5],
standardised with the Sen1Floods11 hand-labelled train mean (−11.19, −17.90) and sd (6.84, 6.58). Loss: masked binary
cross-entropy plus Dice. Stage 1: 12 epochs on 4,384 weak chips, lr 1e-3; stage 2: 30 epochs on 252 hand chips, lr 2e-4.
Augmentation: rotations, flips, per-band offset ±5 dB, contrast ×0.75–1.25 about the chip median, 30% chance of 3 × 3
smoothing in linear power. Seeds 20260927, 1, 2.

### Release v0.1.0 files

| File | SHA-256 |
|---|---|
| `bd_s1_water_checkset_v1.tar.gz` (134 MB: `chips/`, `index.csv`, `qc_decisions.csv`) | `13ae52b3386822c19f34241eeacd92bcd1af4e799ff506129da07d3fc248b145` |
| `unet_s1_water_robust_s20260927.pt` | `63a67c3dac7b48c8ef2a013f3d8a8091c05552d1182142e89d1db4759ac44037` |
| `unet_s1_water_robust_s1.pt` | `6da558652150d6ceff160647611dec9294d4fb9f0c1490a6704d38de85ccde45` |
| `unet_s1_water_robust_s2.pt` | `5889a11b011c2491ba02c766bc761d074068503382baea05a213c7b8b74a16b4` |
| `unet_s1_water.pt` (earlier run, no radiometric augmentation) | `b91dfdbfd8b00c8dcdbfd909cc5fd0657d230a3e3b378a39653084896dd11631` |

### Running locally

Python 3.12. `pip install pystac-client planetary-computer rasterio geopandas shapely scipy scikit-image pandas
segmentation-models-pytorch torch`. The fine-tuning script picks Apple GPU (MPS), CUDA or CPU automatically.

### Data sources and licences

* **Sentinel-1 RTC**: Microsoft Planetary Computer `sentinel-1-rtc` (Catalyst, Microsoft), CC BY 4.0. Contains modified
  Copernicus Sentinel data 2024.
* **Sentinel-2 L2A**: Microsoft Planetary Computer `sentinel-2-l2a`, under the Copernicus Sentinel data terms. Contains
  modified Copernicus Sentinel data 2024.
* **Sen1Floods11 v1.1** (`gs://sen1floods11`), downloaded by notebook 02, not redistributed here. Bonafilia, D.,
  Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning
  flood algorithms for Sentinel-1. CVPR Workshops, 210–211.
* **HydroRIVERS v1.0**, free for non-commercial and commercial use under the HydroSHEDS licence: Lehner, B., Grill, G.
  (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large
  river systems. Hydrological Processes 27(15): 2171–2186. Data at www.hydrosheds.org.
* **JRC Global Surface Water**: Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S. (2016). High-resolution mapping
  of global surface water and its long-term changes. Nature 540: 418–422.
* **Boundaries**: BBS administrative boundaries via OCHA HDX; geoBoundaries IND and MMR ADM0. Downloaded by notebook 01.

Code is MIT licensed (`LICENSE`). The model weights and the check set carry the attribution terms of the data they are
derived from (above).

---

## Glossary

* **Catchment (upland area)**: the land area that drains into a river at a given point; a proxy for river size.
* **Check set**: the 61 hand-checked Bangladesh chips used only for testing and, in held-out halves, for fine-tuning.
* **Chip**: a square cut from a satellite image, here 512 × 512 pixels (about 5 × 5 km).
* **dB (decibel)**: the logarithmic unit radar brightness is reported in; +3 dB is roughly double the returned energy.
* **Dice loss**: a training penalty that rewards overlap between predicted and true water, useful when water is a small
  share of the image.
* **Fine-tuning**: continuing to train an already-trained model on a small new dataset.
* **Fold**: one of the two halves the check set is split into; each half is scored by a model trained on the other.
* **gamma0 / sigma0**: two ways of expressing radar brightness; gamma0 RTC also corrects for terrain slope. The same
  ground gives different numbers under each.
* **ImageNet**: a large collection of ordinary photographs; networks pre-trained on it are a common generic starting
  point.
* **IoU (intersection over union)**: overlap of predicted and true water divided by their union; 1 is perfect. "Pooled"
  means the overlaps and unions are summed over all chips before dividing.
* **L2A**: Sentinel-2 processing level with atmospheric correction ("bottom of atmosphere" reflectance).
* **MNDWI / NDWI**: water indices from two light bands (green with shortwave infrared, or green with near infrared);
  positive values usually mean water.
* **Nearest-neighbour / bilinear reprojection**: ways of re-gridding an image; nearest keeps original pixel values,
  bilinear averages neighbours and smooths the image.
* **Otsu threshold**: an automatic cut-off that best separates an image's pixels into two groups by brightness.
* **Planetary Computer**: Microsoft's free archive of satellite data, searchable through STAC.
* **Release asset**: a large file attached to a tagged version of this repository on GitHub instead of stored in it.
* **ResNet-34**: a standard image-recognition network, used here as the encoder half of the U-Net.
* **RTC (radiometric terrain correction)**: processing that removes brightness caused by slopes facing toward or away
  from the radar.
* **SCL (scene classification layer)**: Sentinel-2's own per-pixel label (water, cloud, shadow, vegetation, …).
* **Seed**: the number that fixes a training run's random choices; different seeds show how much results vary by chance.
* **Sen1Floods11**: the public radar flood dataset used for training (11 flood events worldwide, none in Bangladesh).
* **SHA-256**: a fingerprint of a file; if one byte changes, the fingerprint changes.
* **Speckle**: the grainy texture inherent to radar images.
* **STAC**: a standard catalogue format for searching satellite images by place, time and cloud cover.
* **Strahler order**: a river-size rank; headwater streams are order 1, and order rises where streams of equal order join.
* **U-Net**: a neural network that labels every pixel of an image; here, water or not water.
* **VV / VH**: Sentinel-1's two radar channels (sent vertical, received vertical or horizontal).
* **Weak labels**: training labels made automatically (here from optical images) rather than by a person.
