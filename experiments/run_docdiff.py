#!/usr/bin/env python3
"""DocDiff (ACM MM 2023) on our pages, with its published weights.

    python run_docdiff.py {seal|deblur|<init.pth>:<denoiser.pth>} OUTDIR [--steps N] [--src in|norm] tags...

Mirrors the authors' own inference (demo/inference.ipynb): 128x128 tiles padded
with white, coarse predictor -> deterministic x0-predicting DDIM sampling of the
residual -> final = init + residual. Only change: MPS instead of CUDA, and an
optional step stride (the paper uses all 100).
"""
import argparse, os, sys
import cv2, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD = os.path.join(REPO, "third_party", "docdiff")     # weights: fetch_weights.py
sys.path.insert(0, DD)
from model.DocDiff import DocDiff
from schedule.schedule import Schedule

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(which):
    net = DocDiff(input_channels=6, output_channels=3, n_channels=32,
                  ch_mults=[1, 2, 3, 4], n_blocks=1).to(dev).eval()
    if which == "seal":
        i, d = f"{DD}/checksave/seal_init.pth", f"{DD}/checksave/seal_denoiser.pth"
    elif which == "deblur":
        i, d = f"{DD}/checksave/init.pth", f"{DD}/checksave/denoiser.pth"
    else:
        i, d = which.split(":")
    net.init_predictor.load_state_dict(torch.load(i, map_location=dev))
    net.denoiser.load_state_dict(torch.load(d, map_location=dev))
    return net


@torch.no_grad()
def restore(net, rgb, steps=100, T=100, tile=128, batch=64):
    """rgb uint8 HxWx3 -> (final, init) float [0,1] HxWx3."""
    betas = Schedule("linear", T).get_betas().to(dev).float()
    g = torch.cumprod(1 - betas, 0)
    sg, s1g = g.sqrt(), (1 - g).sqrt()
    H, W = rgb.shape[:2]
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    pad = np.full((Hp, Wp, 3), 255, np.uint8)
    pad[:H, :W] = rgb
    x = torch.from_numpy(pad).permute(2, 0, 1).float().div(255)
    tiles = x.unfold(1, tile, tile).unfold(2, tile, tile)          # 3,ny,nx,t,t
    ny, nx = tiles.shape[1:3]
    tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, 3, tile, tile)
    fin, ini = [], []
    ts = list(range(T - 1, 0, -max(1, T // steps))) + [0]
    for b in range(0, len(tiles), batch):
        c = tiles[b:b + batch].to(dev)
        init = net.init_predictor(c, 0)
        torch.manual_seed(0)
        xt = torch.randn_like(c)
        for k, t in enumerate(ts):
            tt = torch.full((c.shape[0],), t, device=dev, dtype=torch.long)
            ori = net.denoiser(torch.cat((xt, init), 1), tt)
            if t == 0:
                xt = ori
                break
            eps = (xt - sg[t] * ori) / s1g[t]
            tp = ts[k + 1]
            xt = sg[tp] * ori + s1g[tp] * eps
        fin.append((xt + init).cpu()); ini.append(init.cpu())

    def stitch(parts):
        t_ = torch.cat(parts).reshape(ny, nx, 3, tile, tile).permute(2, 0, 3, 1, 4)
        return t_.reshape(3, Hp, Wp)[:, :H, :W].permute(1, 2, 0).clamp(0, 1).numpy()
    return stitch(fin), stitch(ini)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which"); ap.add_argument("out")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--src", default="in")
    a = ap.parse_args()
    net = load(a.which)
    os.makedirs(a.out, exist_ok=True)
    import time
    for tag in a.tags:
        t0 = time.time()
        bgr = cv2.imread(f"{HERE}/{a.src}/{tag}.png", cv2.IMREAD_COLOR)
        fin, ini = restore(net, bgr[:, :, ::-1].copy(), steps=a.steps)
        cv2.imwrite(f"{a.out}/{tag}.png", (fin[:, :, ::-1] * 255 + .5).astype(np.uint8))
        cv2.imwrite(f"{a.out}/{tag}.init.png", (ini[:, :, ::-1] * 255 + .5).astype(np.uint8))
        print(tag, f"{time.time()-t0:.1f}s", flush=True)
