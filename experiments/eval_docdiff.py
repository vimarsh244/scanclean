#!/usr/bin/env python3
"""Score a DocDiff output dir against scanclean on held-out pages.

    python3 eval_docdiff.py OUTDIR tags...

Per page:
  margin_ink   dark px outside the detected local column (lower = cleaner edges)
  lost         scanclean glyph components inside detected lines that DocDiff
               drops (< 30% of their pixels still dark)          -> text lost
  invented     DocDiff components inside detected lines with no ink in the
               ORIGINAL scan under them (< 30% overlap)           -> hallucination
Writes OUTDIR/_diff_<tag>.png: lost in red, invented in blue, over DocDiff.
"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scanclean.core as sc

out, tags = sys.argv[1], sys.argv[2:]


def comps(b):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
    return n, lab, st


for t in tags:
    im = cv2.imread(f"{HERE}/in/{t}.png")
    norm = cv2.imread(f"{HERE}/norm/{t}.png", 0)
    ref = cv2.imread(f"{HERE}/sc/{t}.png", 0)
    dd = cv2.imread(f"{out}/{t}.png", 0)
    dm, boxes = sc.detect_text(im)
    gh = float(np.median([b[3] for b in boxes])) / 2.2
    lines = cv2.dilate(dm, np.ones((int(0.6 * gh) | 1, int(1.2 * gh) | 1), np.uint8)) > 0
    col = np.zeros_like(lines)
    xs = [b[0] for b in boxes if b[2] > 4 * gh]; xe = [b[0] + b[2] for b in boxes if b[2] > 4 * gh]
    col[:, max(0, min(xs) - int(gh)):max(xe) + int(gh)] = True
    R, D, O = ref < 128, dd < 128, norm < 150
    res = {"margin_ink_sc": int((R & ~col).sum()), "margin_ink_dd": int((D & ~col).sum())}
    vis = cv2.cvtColor(dd, cv2.COLOR_GRAY2BGR)
    n, lab, st = comps(R & lines)
    lost = 0
    for j in range(1, n):
        m = lab == j
        if st[j, 4] >= 3 and D[m].mean() < 0.3:
            lost += 1; vis[cv2.dilate(m.astype(np.uint8), np.ones((5, 5))) > 0] = (0, 0, 255)
    n, lab, st = comps(D & lines)
    inv = 0
    for j in range(1, n):
        m = lab == j
        if st[j, 4] >= 3 and O[m].mean() < 0.3:
            inv += 1; vis[cv2.dilate(m.astype(np.uint8), np.ones((5, 5))) > 0] = (255, 80, 0)
    res.update(lost=lost, invented=inv, mean_abs_diff_in_lines=round(float(np.abs(dd.astype(int) - ref)[lines].mean()), 2))
    cv2.imwrite(f"{out}/_diff_{t}.png", vis)
    print(t, res, flush=True)
