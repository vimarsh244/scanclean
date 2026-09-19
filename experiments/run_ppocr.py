#!/usr/bin/env python3
"""PP-OCR text detection (no recognition) -> text polygons per page.

    python run_ppocr.py MODEL OUTDIR tags...     MODEL e.g. PP-OCRv6_tiny_det

Writes OUTDIR/<tag>.png (polygons drawn over the page) and <tag>.npy (filled mask).
"""
import os, sys, time
import cv2, numpy as np
from paddleocr import TextDetection

HERE = os.path.dirname(os.path.abspath(__file__))
model, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
if tags == ["all"]:
    tags = sorted(f[:-4] for f in os.listdir(f"{HERE}/in") if f.endswith(".png"))
os.makedirs(out, exist_ok=True)
det = TextDetection(model_name=model)
for tag in tags:
    img = cv2.imread(f"{HERE}/in/{tag}.png")
    t0 = time.time()
    res = det.predict(img, batch_size=1)
    dt = time.time() - t0
    polys = res[0]["dt_polys"]
    mask = np.zeros(img.shape[:2], np.uint8)
    vis = img.copy()
    for p in polys:
        p = np.asarray(p, np.int32)
        cv2.fillPoly(mask, [p], 255)
        cv2.polylines(vis, [p], True, (0, 0, 255), 2)
    cv2.imwrite(f"{out}/{tag}.png", vis)
    np.save(f"{out}/{tag}.npy", mask)
    np.save(f"{out}/{tag}.polys.npy", np.array([np.asarray(p, np.float32).reshape(-1, 2)[:4] for p in polys]))
    print(tag, len(polys), "boxes", f"{dt*1000:.0f} ms", flush=True)
