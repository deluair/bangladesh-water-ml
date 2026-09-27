"""Step 3: QC PNGs (4-panel per chip) + contact sheets, from cached npz + index.csv."""
import os, glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PANEL = 256
OUT_QC = "qc"

df = pd.read_csv("index.csv")
print(f"Building QC for {len(df)} chips")


def stretch(arr, lo=2, hi=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype="uint8")
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    out = np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)
    out = np.nan_to_num(out, nan=0.0)
    return (out * 255).astype("uint8")


def to_img(r, g, b, size=PANEL):
    rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
    im = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.BILINEAR)
    return im


def vv_gray(vv_db, size=PANEL):
    a = np.clip((vv_db - (-25)) / (0 - (-25)), 0, 1)
    a = np.nan_to_num(a, nan=0.0)
    g = (a * 255).astype("uint8")
    im = Image.fromarray(np.stack([g, g, g], axis=-1), mode="RGB").resize((size, size), Image.BILINEAR)
    return im


def label_img(label, size=PANEL):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype="uint8")
    rgb[label == 0] = (255, 255, 255)
    rgb[label == 1] = (30, 80, 220)
    rgb[label == -1] = (220, 30, 30)
    im = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.NEAREST)
    return im


def build_strip(chip_id):
    npz_path = f"{OUT_QC}/.cache_{chip_id}.npz"
    d = np.load(npz_path)
    b02, b03, b04, b08, b11 = d["b02"], d["b03"], d["b04"], d["b08"], d["b11"]
    vv_db, label = d["vv_db"], d["label"]
    tc = to_img(b04, b03, b02)
    fc = to_img(b11, b08, b04)
    vv = vv_gray(vv_db)
    lb = label_img(label)
    strip = Image.new("RGB", (PANEL * 4, PANEL), (0, 0, 0))
    for i, im in enumerate([tc, fc, vv, lb]):
        strip.paste(im, (i * PANEL, 0))
    return strip


try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
except Exception:
    font = ImageFont.load_default()

ok, fail = 0, 0
strips = {}
for _, row in df.iterrows():
    cid = row["id"]
    try:
        strip = build_strip(cid)
        strip.save(f"{OUT_QC}/{cid}.png")
        strips[cid] = strip
        ok += 1
    except Exception as e:
        print("QC fail for", cid, e)
        fail += 1

print(f"Per-chip QC PNGs: ok={ok} fail={fail}")

# contact sheets, 8 rows per sheet, label id at left
ids = list(df["id"])
ROWS_PER_SHEET = 8
LABEL_W = 140
ROW_H = PANEL
SHEET_W = LABEL_W + PANEL * 4

n_sheets = 0
for s in range(0, len(ids), ROWS_PER_SHEET):
    batch = ids[s:s + ROWS_PER_SHEET]
    sheet = Image.new("RGB", (SHEET_W, ROW_H * len(batch)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, cid in enumerate(batch):
        if cid not in strips:
            continue
        sheet.paste(strips[cid], (LABEL_W, i * ROW_H))
        draw.text((6, i * ROW_H + ROW_H // 2 - 6), cid, fill=(0, 0, 0), font=font)
    sheet_idx = s // ROWS_PER_SHEET
    sheet.save(f"{OUT_QC}/sheet_{sheet_idx:02d}.png")
    n_sheets += 1

print(f"Contact sheets written: {n_sheets}")
print("DONE_STEP3")
