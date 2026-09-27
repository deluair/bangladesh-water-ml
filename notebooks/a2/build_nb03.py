"""Build notebooks/03_bangladesh_checkset_finetune.ipynb (run from the repo root)."""
import nbformat as nbf

REL = "https://github.com/deluair/bangladesh-water-ml/releases/download/v0.1.0"
md = """# 03 · Bangladesh check set: why Sen1Floods11 models fail here, and what fixes it

Stage A2, part 2. Notebook 02 trained Sentinel-1 water U-Nets on [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
(IoU 0.66 on its test split). This notebook tests those models on **61 Bangladesh chips** that Sen1Floods11 does not contain,
then fine-tunes on Bangladesh chips with every chip scored by a model that never saw its tile (or its date).

**Check set** (release asset `bd_s1_water_checkset_v1.tar.gz`): 78 chips of 512 × 512 px, Aug–Oct 2024 flood season.
* Radar: Sentinel-1 RTC gamma0 (VV, VH, dB) from Microsoft Planetary Computer `sentinel-1-rtc`, resampled with nearest neighbour.
* Label: water from a coincident Sentinel-2 L2A scene (4.6–19.5 h apart): MNDWI (B03, B11) > 0 or scene class water;
  no-data, saturated, cloud-shadow, cloud and cirrus pixels (SCL 0, 1, 3, 8, 9, 10) set to -1 and ignored.
* Every chip was inspected by eye; 17 were rejected **only for optical defects** (13 unmasked cumulus, 4 haze with cumulus), never
  because the radar disagreed with the label. Decisions and reasons are in `qc_decisions.csv`; 61 are kept.

**Baseline:** Otsu threshold on VV, fitted per chip (offset-invariant, so it is immune to calibration shifts).

Runs on any Colab GPU in about 10 minutes."""

cells = [
    nbf.v4.new_markdown_cell(md),
    nbf.v4.new_code_cell("""!pip -q install segmentation-models-pytorch rasterio scikit-image
import os, hashlib, time, urllib.request, tarfile, numpy as np, pandas as pd, rasterio, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU only')"""),
    nbf.v4.new_code_cell(f"""# Release assets, checked against their sha256
REL = '{REL}'
ASSETS = {{
    'bd_s1_water_checkset_v1.tar.gz': '13ae52b3386822c19f34241eeacd92bcd1af4e799ff506129da07d3fc248b145',
    'unet_s1_water.pt': 'b91dfdbfd8b00c8dcdbfd909cc5fd0657d230a3e3b378a39653084896dd11631',
    'unet_s1_water_robust_s20260927.pt': '63a67c3dac7b48c8ef2a013f3d8a8091c05552d1182142e89d1db4759ac44037',
    'unet_s1_water_robust_s1.pt': '6da558652150d6ceff160647611dec9294d4fb9f0c1490a6704d38de85ccde45',
    'unet_s1_water_robust_s2.pt': '5889a11b011c2491ba02c766bc761d074068503382baea05a213c7b8b74a16b4',
}}
for f, h in ASSETS.items():
    if not os.path.exists(f): urllib.request.urlretrieve(f'{{REL}}/{{f}}', f)
    assert hashlib.sha256(open(f, 'rb').read()).hexdigest() == h, f
if not os.path.exists('bd/index.csv'):
    os.makedirs('bd', exist_ok=True); tarfile.open('bd_s1_water_checkset_v1.tar.gz').extractall('bd')
q = pd.read_csv('bd/qc_decisions.csv')
print(q.decision.value_counts().to_dict()); print(q[q.decision != 'keep'].reason.value_counts().to_dict())"""),
    nbf.v4.new_code_cell("""# Kept chips; normalisation = Sen1Floods11 hand-labelled train split (notebook 02)
MU = np.array([-11.188685, -17.898247], 'float32'); SD = np.array([6.83887, 6.5755134], 'float32')
q = q[q.decision == 'keep'].reset_index(drop=True)
def rd(p):
    with rasterio.open(p) as d: return d.read()
X = np.stack([np.nan_to_num(np.clip(rd(f'bd/chips/{i}_S1.tif').astype('float32'), -50, 5), nan=-50.0) for i in q.id])
Y = np.stack([rd(f'bd/chips/{i}_Label.tif')[0].astype('int64') for i in q.id])
nrm = lambda x: (x - MU[:, None, None]) / SD[:, None, None]
print(X.shape, 'water share of labelled pixels', round(float((Y == 1).sum() / (Y >= 0).sum()), 3))

def predict(model, idx):
    model.eval(); out = []
    with torch.no_grad():
        for i in idx: out.append((model(torch.from_numpy(nrm(X[i])[None]).to(dev))[0, 0] > 0).cpu().numpy())
    return out
def pooled(preds, idx):
    I = U = 0
    for p, i in zip(preds, idx):
        v = Y[i] >= 0; t = Y[i] == 1; I += (p & t & v).sum(); U += ((p | t) & v).sum()
    return I / U
oI = oU = 0
for i in range(len(q)):
    vv = X[i][0]; v = (Y[i] >= 0) & (vv > -50); t = Y[i] == 1; po = (vv < threshold_otsu(vv[v])) & v
    oI += (po & t & v).sum(); oU += ((po | t) & v).sum()
OTSU = oI / oU; print(f'Otsu (per-chip VV threshold) pooled IoU {OTSU:.3f}')"""),
    nbf.v4.new_code_cell("""# 1. Sen1Floods11 models applied to Bangladesh as they are
def load(f):
    m = smp.Unet('resnet34', encoder_weights=None, in_channels=2, classes=1)
    m.load_state_dict(torch.load(f, map_location='cpu')); return m.to(dev)
allidx = np.arange(len(q)); rows = []
for f in ['unet_s1_water.pt'] + [f'unet_s1_water_robust_s{s}.pt' for s in (20260927, 1, 2)]:
    rows.append(dict(model=f, bd_pooled_iou=round(pooled(predict(load(f), allidx), allidx), 3)))
zero_shot = pd.DataFrame(rows); zero_shot.loc[len(zero_shot)] = ['Otsu baseline', round(OTSU, 3)]; zero_shot"""),
    nbf.v4.new_code_cell("""# 2. Fine-tune on Bangladesh chips, 2-fold grouped split: each chip is scored by a model that never saw its group
def aug(x, y):
    med = np.median(x, axis=(1, 2), keepdims=True)
    x = (x - med) * np.random.uniform(0.75, 1.25, (2, 1, 1)) + med + np.random.uniform(-5, 5, (2, 1, 1))
    if np.random.rand() < .3: x = 10 * np.log10(np.maximum(uniform_filter(10 ** (x / 10), (1, 3, 3)), 1e-6))
    k = np.random.randint(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if np.random.rand() < .5: x, y = x[:, :, ::-1], y[:, ::-1]
    return x.astype('float32').copy(), y.copy()
def loss_fn(lg, y):
    m = (y >= 0).float(); t = y.clamp(min=0).float(); lg = lg[:, 0]
    bce = (F.binary_cross_entropy_with_logits(lg, t, reduction='none') * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(lg) * m; return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
def folds(split):
    g = q.id.str.extract(r'_(T\\d\\d[A-Z]{3})_')[0] if split == 'tile' else q.id.str[3:11]   # S2 tile, or S2 date
    fold, load_ = {}, [0, 0]
    for k, n in g.value_counts().items(): j = int(np.argmin(load_)); fold[k] = j; load_[j] += n   # greedy balance
    return g.map(fold).values, g.nunique()
res = []
for split in ('tile', 'date'):
    fd, ng = folds(split)
    for init in ('s1f11', 'imagenet'):
        for seed in (20260927, 1, 2):
            preds, idxs = [], []
            for f in (0, 1):
                torch.manual_seed(seed); np.random.seed(seed)
                tr, te = np.where(fd != f)[0], np.where(fd == f)[0]
                m = (smp.Unet('resnet34', encoder_weights='imagenet', in_channels=2, classes=1).to(dev) if init == 'imagenet'
                     else load(f'unet_s1_water_robust_s{seed}.pt'))
                opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4)
                for ep in range(25):
                    m.train(); perm = np.random.permutation(tr)
                    for b in range(0, len(perm), 4):
                        xb, yb = zip(*[aug(X[i], Y[i]) for i in perm[b:b + 4]])
                        l = loss_fn(m(torch.from_numpy(nrm(np.stack(xb))).to(dev)), torch.from_numpy(np.stack(yb)).to(dev))
                        opt.zero_grad(); l.backward(); opt.step()
                preds += predict(m, te); idxs += list(te)
            res.append(dict(split=split, groups=ng, init=init, seed=seed, iou=pooled(preds, idxs)))
            print(res[-1])
res = pd.DataFrame(res); res.to_csv('bd_finetune_colab.csv', index=False)"""),
    nbf.v4.new_code_cell("""summary = res.groupby(['split', 'init']).iou.agg(['mean', 'std', 'count']).round(3)
summary['otsu'] = round(OTSU, 3); summary"""),
    nbf.v4.new_markdown_cell("""**Reading the result.** The Sen1Floods11 models transfer badly to Planetary Computer RTC chips over Bangladesh: the
radar product is calibrated differently (gamma0 RTC, not the sigma0 GEE product Sen1Floods11 uses; over Bangladesh land
it reads about 3.5 dB (VV) and 5 dB (VH) brighter) and the model leans on speckle texture. Otsu survives because a per-chip threshold
ignores calibration offsets. About 30 Bangladesh chips per fold are enough to lift the U-Net well past Otsu; whether the
encoder starts from Sen1Floods11 or ImageNet changes the result by about 0.01, so the local labels do the work."""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "notebooks/03_bangladesh_checkset_finetune.ipynb")
