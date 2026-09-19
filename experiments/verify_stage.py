#!/usr/bin/env python3
"""Contact sheet of everything one stage removed (or rescued), all pages.

    python3 verify_stage.py gate|rescued|... [out_prefix]
Each tile: original (straightened) with the stage's pixels outlined in red |
the final output. Regions are grouped within one glyph height.
"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scanclean as sc
from audit_lines import Opt


class O(Opt):
    black, white, soften, sharpen, upscale, bilevel = 40, 224, 1.0, 0.8, 1, False
    paper_floor, audit = True, False


stage = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/out/_v_{stage}"
tiles = []
for s in (1, 2, 3, 4):
    for i, bgr, dpi in sc.page_images(f"{os.path.dirname(HERE)}/original/Sample {s}.pdf"):
        tag = f"s{s}p{i+1:02d}"
        a = sc.analyse(bgr, dpi, O)
        out = sc.clean_page(bgr, dpi, O)[0]
        straight, _ = sc.straighten(bgr, dpi / 300)
        m, gh = a["stages"][stage], a["gh"]
        grp = cv2.dilate(m, np.ones((int(gh), int(gh)), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
        for j in range(1, n):
            x, y, w, h, _ = st[j]
            p = int(1.2 * gh)
            y0, y1 = max(0, y - p), min(m.shape[0], y + h + p)
            x0, x1 = max(0, x - 2 * p), min(m.shape[1], x + w + 2 * p)
            A = straight[y0:y1, x0:x1].copy()
            cs, _ = cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(A, cs, -1, (0, 0, 255), 1)
            B = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            t = np.hstack([A, np.full((A.shape[0], 3, 3), 128, np.uint8), B])
            sf = min(2.0, 460 / t.shape[1], 300 / t.shape[0])
            t = cv2.resize(t, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA if sf < 1 else cv2.INTER_CUBIC)
            cv2.putText(t, f"{tag} ({x},{y})", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 0, 0), 1)
            tiles.append(t)
print(len(tiles), "regions")
if tiles:
    W, H = 470, 310
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, W - t.shape[1], cv2.BORDER_CONSTANT, value=(225, 225, 225)) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 225))
    sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    for q in range(0, sheet.shape[0], 5 * H):
        cv2.imwrite(f"{prefix}_{q // (5 * H)}.png", sheet[q:q + 5 * H])
    print((sheet.shape[0] + 5 * H - 1) // (5 * H), "sheets")
