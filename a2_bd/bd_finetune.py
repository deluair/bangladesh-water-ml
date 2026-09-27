"""Fine-tune the robust Sen1Floods11 U-Nets on Bangladesh chips, 2-fold split by Sentinel-2 tile (each chip scored once, by a model that never saw its tile)."""
import os, numpy as np, pandas as pd, rasterio, torch, torch.nn.functional as F, segmentation_models_pytorch as smp, time
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
MU = np.array([-11.188685, -17.898247], 'float32'); SD = np.array([6.83887, 6.5755134], 'float32')
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
q = pd.read_csv('qc_decisions.csv'); q = q[q.decision == 'keep'].reset_index(drop=True)
if os.path.exists('dedupe_dropped.csv'):  # v2: chips that are the same ESA acquisition reprocessed, see step2c_dedupe.py
    dropped = set(pd.read_csv('dedupe_dropped.csv').id); q = q[~q.id.isin(dropped)].reset_index(drop=True)
q['tile'] = q.id.str.extract(r'_(T\d\d[A-Z]{3})_')[0]
SPLIT, INIT = os.environ.get("SPLIT", "tile"), os.environ.get("INIT", "s1f11")
if SPLIT == "date": q["tile"] = q.id.str[3:11]   # group by S2 acquisition date instead of tile
tiles = q.groupby('tile').size().sort_values(ascending=False)
fold, load = {}, [0, 0]
for t, n in tiles.items(): k = int(np.argmin(load)); fold[t] = k; load[k] += n   # greedy balance by chip count
q['fold'] = q.tile.map(fold)
print('split', SPLIT, 'init', INIT, 'device', dev, '| tiles', len(tiles), '| chips per fold', q.fold.value_counts().sort_index().tolist())
rd = lambda p: rasterio.open(p).read()
X = np.stack([np.nan_to_num(np.clip(rd(f'chips/{i}_S1.tif').astype('float32'), -50, 5), nan=-50.0) for i in q.id])
Y = np.stack([rd(f'chips/{i}_Label.tif')[0].astype('int64') for i in q.id])
def aug(x, y):
    med = np.median(x, axis=(1, 2), keepdims=True)
    x = (x - med) * np.random.uniform(0.75, 1.25, (2, 1, 1)) + med + np.random.uniform(-5, 5, (2, 1, 1))
    if np.random.rand() < .3: x = 10 * np.log10(np.maximum(uniform_filter(10 ** (x / 10), (1, 3, 3)), 1e-6))
    k = np.random.randint(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if np.random.rand() < .5: x, y = x[:, :, ::-1], y[:, ::-1]
    return x.astype('float32').copy(), y.copy()
nrm = lambda x: (x - MU[:, None, None]) / SD[:, None, None]
def loss_fn(lg, y):
    m = (y >= 0).float(); t = y.clamp(min=0).float(); lg = lg[:, 0]
    bce = (F.binary_cross_entropy_with_logits(lg, t, reduction='none') * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(lg) * m; return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
def predict(model, idx):
    model.eval(); out = []
    with torch.no_grad():
        for i in idx: out.append((model(torch.from_numpy(nrm(X[i])[None]).to(dev))[0, 0] > 0).cpu().numpy())
    return out
rows = []
for seed in [20260927, 1, 2]:
    for f in (0, 1):
        torch.manual_seed(seed); np.random.seed(seed)
        tr, te = np.where(q.fold != f)[0], np.where(q.fold == f)[0]
        if INIT == "imagenet": m = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=2, classes=1).to(dev)
        else:
            m = smp.Unet('resnet34', encoder_weights=None, in_channels=2, classes=1)
            m.load_state_dict(torch.load(os.path.join(os.environ.get('WEIGHTS', '../models'), f'unet_s1_water_robust_s{seed}.pt'), map_location='cpu')); m.to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4); t0 = time.time()
        for ep in range(25):
            m.train(); perm = np.random.permutation(tr)
            for b in range(0, len(perm), 4):
                xb, yb = zip(*[aug(X[i], Y[i]) for i in perm[b:b + 4]])
                l = loss_fn(m(torch.from_numpy(nrm(np.stack(xb))).to(dev)), torch.from_numpy(np.stack(yb)).to(dev))
                opt.zero_grad(); l.backward(); opt.step()
        for i, p in zip(te, predict(m, te)):
            y = Y[i]; v = y >= 0; t = y == 1
            rows.append(dict(seed=seed, fold=f, id=q.id[i], I=int((p & t & v).sum()), U=int(((p | t) & v).sum())))
        print(f'seed {seed} fold {f}: trained on {len(tr)} chips, tested on {len(te)}, {time.time()-t0:.0f}s')
r = pd.DataFrame(rows); r.to_csv(f'bd_finetune_results_{SPLIT}_{INIT}.csv', index=False)
oI = oU = 0
for i in range(len(q)):
    x, y = X[i][0], Y[i]; v = (y >= 0) & (x > -50); t = y == 1; po = (x < threshold_otsu(x[v])) & v
    oI += (po & t & v).sum(); oU += ((po | t) & v).sum()
per = r.groupby('seed').apply(lambda d: d.I.sum() / d.U.sum())
print(f'pooled IoU per seed (all {len(q)} chips, each scored out-of-{SPLIT}):', per.round(3).to_dict())
print(f'fine-tuned U-Net mean {per.mean():.3f} sd {per.std():.3f} | Otsu {oI/oU:.3f}')
