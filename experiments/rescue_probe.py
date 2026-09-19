import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import scanclean as sc
from audit_lines import Opt
from textdet import detect
tiles = []
for s in (1, 2, 3, 4):
    for i, bgr, dpi in sc.page_images(f"{os.path.dirname(HERE)}/original/Sample {s}.pdf"):
        tag = f"s{s}p{i+1:02d}"
        a = sc.analyse(bgr, dpi, Opt)
        straight, _ = sc.straighten(bgr, dpi / 300)
        dm, boxes = detect(straight)
        norm, gh = a["norm"], a["gh"]
        st_ = a["stages"]
        R = st_["specks"] | st_["band"] | st_["edge"]
        stamp = cv2.dilate(st_["stamp"], np.ones((int(gh), int(gh)), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(R, 8)
        for j in range(1, n):
            x, y, w, h, ar = st[j]
            cx, cy = int(cen[j][0]), int(cen[j][1])
            if not dm[cy, cx] or w > 0.7 * gh or h > 0.7 * gh:
                continue
            comp = (lab == j).astype(np.uint8)
            mn = int(norm[comp > 0].min())
            if mn >= 90:
                continue
            r = max(3, int(0.4 * gh))
            ring = cv2.dilate(comp, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) & ~cv2.dilate(comp, np.ones((5, 5), np.uint8))
            ringmed = int(np.median(norm[ring > 0]))
            ringp10 = int(np.percentile(norm[ring > 0], 10))
            nearstamp = bool(stamp[cy, cx])
            pr = detect.prob
            k = max(3, int(0.15 * gh)) | 1
            pmax = float(cv2.dilate(pr, np.ones((k, k), np.float32))[comp > 0].max())
            pmean = float(pr[comp > 0].mean())
            print(f"{tag} ({x},{y}) {w}x{h} min{mn} prob_max{pmax:.2f} prob_mean{pmean:.2f}")
