import sys, os, numpy as np, pandas as pd, rasterio, torch, segmentation_models_pytorch as smp
from skimage.filters import threshold_otsu
MU = np.array([-11.188685, -17.898247], 'float32'); SD = np.array([6.83887, 6.5755134], 'float32')  # train-split stats from the Colab run
q = pd.read_csv(sys.argv[1]); ids = q[q.decision == 'keep'].id.tolist()
dd = os.path.join(os.path.dirname(sys.argv[1]) or '.', 'dedupe_dropped.csv')  # v2: chips that are the same ESA acquisition reprocessed, see step2c_dedupe.py
if os.path.exists(dd):
    dropped = set(pd.read_csv(dd).id); ids = [i for i in ids if i not in dropped]
m = smp.Unet('resnet34', encoder_weights=None, in_channels=2, classes=1)
m.load_state_dict(torch.load(sys.argv[2], map_location='cpu')); m.eval()
def rd(p):
    with rasterio.open(p) as d: return d.read()
rows = []; I = U = oI = oU = 0
for i in ids:
    x = np.nan_to_num(np.clip(rd(f'chips/{i}_S1.tif').astype('float32'), -50, 5), nan=-50.0); y = rd(f'chips/{i}_Label.tif')[0]
    with torch.no_grad(): p = (m(torch.from_numpy(((x - MU[:, None, None]) / SD[:, None, None])[None]))[0, 0] > 0).numpy()
    v = y >= 0; t = y == 1
    vv = x[0]; vo = v & (vv > -50); th = threshold_otsu(vv[vo]); po = (vv < th) & vo
    a, b = (p & t & v).sum(), ((p | t) & v).sum(); c, d = (po & t & vo).sum(), ((po | t) & vo).sum()
    I += a; U += b; oI += c; oU += d
    rows.append(dict(id=i, water_frac=round(t[v].mean(), 3), unet_iou=round(a / b, 3) if b else np.nan, otsu_iou=round(c / d, 3) if d else np.nan,
                unet_I=int(a), unet_U=int(b), otsu_I=int(c), otsu_U=int(d)))
r = pd.DataFrame(rows); r.to_csv(sys.argv[3], index=False)
print(f'chips {len(r)}  pooled IoU unet {I/U:.3f} otsu {oI/oU:.3f}  chip-mean unet {r.unet_iou.mean():.3f} otsu {r.otsu_iou.mean():.3f}')
print(r.sort_values('unet_iou').to_string(index=False))
