#!/usr/bin/env python3
"""Download DocDiff's published weights (64 MB, not kept in git) into checksave/.

    python3 third_party/docdiff/fetch_weights.py
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/Royalvice/DocDiff/main/checksave/"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksave")
os.makedirs(DST, exist_ok=True)
for f in ("init.pth", "denoiser.pth", "seal_init.pth", "seal_denoiser.pth"):
    path = os.path.join(DST, f)
    if not os.path.exists(path):
        print("fetching", f)
        urllib.request.urlretrieve(URL + f, path)
print("weights in", DST)
