import nbformat as nbf
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# 02 · Flood water from radar: training a Sentinel-1 U-Net on Sen1Floods11

Stage A2 of **bangladesh-water-ml**. Train a water-segmentation model on Sentinel-1 radar (VV, VH) from
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., CVPR-W 2020; public bucket `gs://sen1floods11`, v1.1),
and test it on Sen1Floods11's held-out splits. Notebook 03 tests these weights on Bangladesh.

* **Train:** 4,384 weakly labelled chips (S1Weak + S2IndexLabelWeak), then fine-tune on the 252 hand-labelled train chips.
* **Test:** Sen1Floods11 hand-labelled test (90) and Bolivia hold-out (15).
* **Augmentation:** per-band offset (±5 dB) and contrast (0.75–1.25) plus occasional 3 × 3 smoothing, so the model does not depend on one radar calibration.
* **Baseline:** per-chip Otsu threshold on VV, the standard non-ML method. The model has to beat it to be worth using.

Runtime: GPU. The published weights come from a Colab A100 run on 2026-09-27 (3 seeds, about 50 min); results in `models/a2_results_robust_s1f11.csv`.""")
code("""!pip -q install segmentation-models-pytorch rasterio scikit-image
import os, glob, time, numpy as np, pandas as pd, rasterio, torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')""")
code("""# Sen1Floods11 v1.1 from the public bucket (no login needed)
B = 'gs://sen1floods11/v1.1'
os.makedirs('s1f11', exist_ok=True)
if not os.path.exists('s1f11/done'):
    !gsutil -m -q cp -r {B}/data/flood_events/HandLabeled/S1Hand {B}/data/flood_events/HandLabeled/LabelHand s1f11/
    !gsutil -m -q cp -r {B}/data/flood_events/WeaklyLabeled/S1Weak {B}/data/flood_events/WeaklyLabeled/S2IndexLabelWeak s1f11/
    !gsutil -q cp -r {B}/splits/flood_handlabeled s1f11/
    open('s1f11/done', 'w').close()
for d in ['S1Hand', 'LabelHand', 'S1Weak', 'S2IndexLabelWeak']:
    print(d, len(glob.glob(f's1f11/{d}/*.tif')))""")
code("""def read_s1(p):
    with rasterio.open(p) as d: a = d.read().astype('float32')
    return np.nan_to_num(np.clip(a, -50, 5), nan=-50.0)
def read_lab(p, weak=False):
    with rasterio.open(p) as d: y = d.read(1).astype('int16')
    if weak: y = np.where(y > 0, 1, np.where(y == 0, 0, -1)).astype('int16')  # weak labels: 1 water, 0 dry, else unknown
    return y
def split(name):
    df = pd.read_csv(f's1f11/flood_handlabeled/flood_{name}_data.csv', header=None)
    return [(f's1f11/S1Hand/{a}', f's1f11/LabelHand/{b}') for a, b in df.values]
hand = {k: split(k) for k in ['train', 'valid', 'test', 'bolivia']}
weak = [(p, p.replace('S1Weak', 'S2IndexLabelWeak')) for p in sorted(glob.glob('s1f11/S1Weak/*.tif'))]
weak = [w for w in weak if os.path.exists(w[1])]
print({k: len(v) for k, v in hand.items()}, 'weak', len(weak))
y0 = read_lab(weak[0][1]); print('weak label values', np.unique(y0))""")
code("""# Normalisation from the hand-labelled train split only
xs = np.stack([read_s1(p) for p, _ in hand['train']])
MU, SD = xs.mean((0, 2, 3)), xs.std((0, 2, 3)); del xs
print('VV/VH mean dB', MU, 'sd', SD)

class DS(torch.utils.data.Dataset):
    def __init__(self, pairs, weak=False, aug=False):
        self.X = np.stack([read_s1(p) for p, _ in pairs]).astype('float16')
        self.Y = np.stack([read_lab(q, weak) for _, q in pairs])
        self.aug = aug
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x, y = self.X[i].astype('float32'), self.Y[i].copy()
        if self.aug:
            # radiometric robustness: Planetary Computer RTC (gamma0) chips over Bangladesh sit ~3.5 dB (VV) and
            # ~5 dB (VH) brighter than Sen1Floods11's GEE sigma0, and a model without this collapsed there (IoU 0.01)
            med = np.median(x, axis=(1, 2), keepdims=True)
            x = (x - med) * np.random.uniform(0.75, 1.25, (2, 1, 1)) + med + np.random.uniform(-5, 5, (2, 1, 1))
            if np.random.rand() < .3:                          # mild speckle smoothing, in linear power
                x = 10 * np.log10(np.maximum(uniform_filter(10 ** (x / 10), (1, 3, 3)), 1e-6))
            x = x.astype('float32')
            k = np.random.randint(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < .5: x, y = x[:, :, ::-1], y[:, ::-1]
        x = (x - MU[:, None, None]) / SD[:, None, None]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy().astype('int64'))""")
code("""import segmentation_models_pytorch as smp, torch.nn.functional as F
from scipy.ndimage import uniform_filter
dev = 'cuda'
def loss_fn(logit, y):
    m = (y >= 0).float(); t = y.clamp(min=0).float(); logit = logit[:, 0]
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction='none') * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
    return bce + dice

def iou_counts(model, pairs, bs=16):
    ds = DS(pairs); dl = torch.utils.data.DataLoader(ds, bs)
    I = U = 0; per = []
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for x, y in dl:
            p = (model(x.to(dev))[:, 0] > 0).cpu(); v = y >= 0; t = y == 1
            for pi, ti, vi in zip(p, t, v):
                i, u = (pi & ti & vi).sum().item(), ((pi | ti) & vi).sum().item()
                I += i; U += u; per.append(i / u if u else np.nan)
    return I / U, np.nanmean(per)

def fit(model, train_pairs, weak, epochs, lr, val_pairs):
    dl = torch.utils.data.DataLoader(DS(train_pairs, weak, aug=True), 16, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler(); best = (-1, None)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0
        for x, y in dl:
            with torch.autocast('cuda', dtype=torch.float16): l = loss_fn(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += l.item()
        v = iou_counts(model, val_pairs)[0]
        if v > best[0]: best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()})
        print(f'ep {ep+1}/{epochs} loss {tl/len(dl):.3f} val IoU {v:.3f} {time.time()-t0:.0f}s')
    model.load_state_dict(best[1]); return best[0]""")
code("""SEEDS = [20260927, 1, 2]
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=2, classes=1).to(dev)
    print(f'seed {seed} stage 1: weak labels'); fit(model, weak, True, 12, 1e-3, hand['valid'])
    print(f'seed {seed} stage 2: hand labels'); fit(model, hand['train'], False, 30, 2e-4, hand['valid'])
    torch.save(model.state_dict(), f'unet_s1_water_robust_s{seed}.pt')""")
code("""from skimage.filters import threshold_otsu
def otsu_counts(pairs):
    I = U = 0; per = []
    for p, q in pairs:
        vv = read_s1(p)[0]; y = read_lab(q); v = (y >= 0) & (vv > -50)
        t = threshold_otsu(vv[v]) if v.sum() > 100 else -18.0
        pr = (vv < t) & v; tr = (y == 1) & v
        i, u = (pr & tr).sum(), (pr | tr).sum(); I += i; U += u; per.append(i / u if u else np.nan)
    return I / U, np.nanmean(per)

rows = []
for name, pairs in [('Sen1Floods11 test', hand['test']), ('Bolivia hold-out', hand['bolivia'])]:
    if not pairs: continue
    oi, op = otsu_counts(pairs)
    for seed in SEEDS:
        model.load_state_dict(torch.load(f'unet_s1_water_robust_s{seed}.pt')); mi, mp = iou_counts(model, pairs)
        rows.append(dict(set=name, seed=seed, chips=len(pairs), unet_iou=round(mi, 3), otsu_iou=round(oi, 3),
                         unet_chip_mean_iou=round(mp, 3), otsu_chip_mean_iou=round(op, 3)))
res = pd.DataFrame(rows); res.to_csv('a2_results.csv', index=False); res""")
nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, 'notebooks/02_flood_radar_unet.ipynb')  # run from the repo root
