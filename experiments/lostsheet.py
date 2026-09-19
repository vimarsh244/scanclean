#!/usr/bin/env python3
"""Contact sheet of every place scanclean deleted ink INSIDE a detected text box.
Each tile: original (flattened) | scanclean, deleted pixels red on the original."""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate import det_mask, tags

tiles = []
for tag in tags:
    kill = cv2.imread(f"{HERE}/kill/{tag}.png", 0)
    norm = cv2.imread(f"{HERE}/norm/{tag}.png", 0)
    sc = cv2.imread(f"{HERE}/sc/{tag}.png", 0)
    dm, lh = det_mask(tag)
    lost = ((kill > 0) & (dm > 0) & (norm < 170)).astype(np.uint8)
    grp = cv2.dilate(lost, np.ones((int(lh), int(lh)), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        px = int(lost[lab == i].sum())
        if px < 25:
            continue
        p = int(lh * 0.6)
        y0, y1 = max(0, y - p), min(norm.shape[0], y + h + p)
        x0, x1 = max(0, x - 3 * p), min(norm.shape[1], x + w + 3 * p)
        a = cv2.cvtColor(norm[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        a[lost[y0:y1, x0:x1] > 0] = (0, 0, 255)
        b = cv2.cvtColor(sc[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        t = np.hstack([a, np.full((a.shape[0], 4, 3), 128, np.uint8), b])
        s = min(2.0, 700 / t.shape[1], 300 / t.shape[0])
        t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{tag} {px}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1)
        tiles.append((px, tag, t))
tiles.sort(key=lambda r: (r[1], -r[0]))
print(len(tiles), "regions")
W = 720
rows, row, rw = [], [], 0
for _, _, t in tiles:
    if rw + t.shape[1] > W * 2 and row:
        rows.append(row); row, rw = [], 0
    row.append(t); rw += t.shape[1] + 6
rows.append(row)
out = []
for r in rows:
    h = max(t.shape[0] for t in r)
    line = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(230, 230, 230)) for t in r])
    out.append(cv2.copyMakeBorder(line, 0, 6, 0, W * 2 + 60 - line.shape[1], cv2.BORDER_CONSTANT, value=(230, 230, 230)))
sheet = np.vstack(out)
for k in range(0, sheet.shape[0], 1600):
    cv2.imwrite(f"{HERE}/out/_lost_{k//1600}.png", sheet[k:k + 1600])
print(sheet.shape)
