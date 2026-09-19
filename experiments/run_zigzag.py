#!/usr/bin/env python3
"""ZigZag (Bloechle et al., DocEng 2024) on our pages - authors' own Python port
(AGPL-3.0, vendored in third_party/zigzag; comparison only, not used by scanclean).
    python3 run_zigzag.py OUTDIR SIZE WEIGHT tags|all"""
import os, sys, time
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "zigzag"))
import zigzag
out, size, weight, tags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
if tags == ["all"]:
    tags = sorted(f[:-4] for f in os.listdir(f"{HERE}/in") if f.endswith(".png"))
os.makedirs(out, exist_ok=True)
for t in tags:
    rgb = cv2.imread(f"{HERE}/in/{t}.png")[:, :, ::-1].copy()
    t0 = time.time()
    g, info = zigzag.process(rgb, mode="gray", size=size, weight=weight)
    b, _ = zigzag.process(rgb, mode="binary", size=size, weight=weight, upsample=False)
    cv2.imwrite(f"{out}/{t}.png", g); cv2.imwrite(f"{out}/{t}.bin.png", b)
    print(t, f"{(time.time()-t0)*1000:.0f} ms", info["threshold"], flush=True)
