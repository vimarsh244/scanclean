#!/usr/bin/env python3
"""Per stage: killed components whose centre lies inside a RAW detector box."""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scanclean.core as sc
from scanclean.cli import page_images
from audit_lines import Opt
det = sys.argv[1] if len(sys.argv) > 1 else "ppocr_tiny"
tot, tiles = {}, []
for s in (1, 2, 3, 4):
    for i, bgr, dpi in page_images(f"{os.path.dirname(HERE)}/original/Sample {s}.pdf"):
        tag = f"s{s}p{i+1:02d}"
        a = sc.analyse(bgr, dpi, Opt)
        raw = np.load(f"{HERE}/out/{det}/{tag}.npy")
        norm = a["norm"]
        im = cv2.imread(f"{HERE}/in/{tag}.png")
        for k, m in a["stages"].items():
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            for j in range(1, n):
                cx, cy = int(cen[j][0]), int(cen[j][1])
                if not raw[cy, cx]:
                    continue
                x, y, w, h, ar = st[j]
                mn = int(norm[lab == j].min())
                tot.setdefault(k, []).append((tag, x, y, w, h, ar, mn))
                if (k == "specks" and mn < 90) or (k in ("band", "edge", "blot", "dust") and mn < 60):
                    r = 40
                    y0, x0 = max(0, cy - r), max(0, cx - 2 * r)
                    A = im[y0:cy + r, x0:cx + 2 * r].copy()
                    cv2.rectangle(A, (x - x0 - 3, y - y0 - 3), (x - x0 + w + 2, y - y0 + h + 2), (0, 0, 255), 1)
                    A = cv2.resize(A, (480, 240), interpolation=cv2.INTER_CUBIC)
                    cv2.putText(A, f"{tag} {k} {w}x{h} min{mn}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1)
                    tiles.append(A)
for k, v in tot.items():
    print(f"{k:8s} {len(v):4d} comps in raw boxes; ink-black(min<60): {sum(1 for r in v if r[6] < 60)}")
if tiles:
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 230))
    sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    for p in range(0, sheet.shape[0], 1680):
        cv2.imwrite(f"{HERE}/out/_specks_{p//1680}.png", sheet[p:p + 1680])
    print(len(tiles), "speck tiles")
