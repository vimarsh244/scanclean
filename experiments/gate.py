#!/usr/bin/env python3
"""What would a text-detector mask change on top of scanclean?

For every page:
  RED   = ink scanclean KEPT that touches no detected text box  -> extra removals
  BLUE  = ink scanclean DELETED inside a detected text box       -> possible lost text
Detector mask = union of PP-OCRv6 tiny + PP-OCRv5 server boxes, dilated.
"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tags = sorted(f[:-4] for f in os.listdir(f"{HERE}/in") if f.endswith(".png"))
os.makedirs(f"{HERE}/out/gate", exist_ok=True)
dets = sys.argv[1:] or ["ppocr_tiny", "ppocr_v5s"]


def det_mask(tag):
    m = None
    for d in dets:
        x = np.load(f"{HERE}/out/{d}/{tag}.npy")
        m = x if m is None else m | x
    # median box height -> line scale
    P = np.load(f"{HERE}/out/{dets[0]}/{tag}.polys.npy")
    lh = float(np.median(P[:, :, 1].max(1) - P[:, :, 1].min(1))) if len(P) else 20.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.6 * lh) | 1, int(0.5 * lh) | 1))
    return cv2.dilate(m, k), lh


def main():
    for tag in tags:
        sc = cv2.imread(f"{HERE}/sc/{tag}.png", 0)
        kill = cv2.imread(f"{HERE}/kill/{tag}.png", 0)
        norm = cv2.imread(f"{HERE}/norm/{tag}.png", 0)
        dm, lh = det_mask(tag)
        ink = (sc < 170).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        hit = np.zeros(n, bool)
        hit[np.unique(lab[dm > 0])] = True
        extra = (~hit[lab]) & (lab > 0)
        lost = (kill > 0) & (dm > 0) & (norm < 170)
        vis = cv2.cvtColor(sc, cv2.COLOR_GRAY2BGR)
        vis[cv2.dilate(extra.astype(np.uint8), np.ones((5, 5))) > 0] = (0, 0, 255)
        vis[cv2.dilate(lost.astype(np.uint8), np.ones((3, 3))) > 0] = (255, 120, 0)
        cnt = [c for c in cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
        cv2.drawContours(vis, cnt, -1, (0, 170, 0), 1)
        cv2.imwrite(f"{HERE}/out/gate/{tag}.png", vis)
        ncomp = int(len(np.unique(lab[extra])))
        print(f"{tag} line_h={lh:.0f} extra: {ncomp:4d} comps {int(extra.sum()):6d} px | "
              f"deleted-inside-boxes: {int(lost.sum()):5d} px", flush=True)


if __name__ == "__main__":
    main()
