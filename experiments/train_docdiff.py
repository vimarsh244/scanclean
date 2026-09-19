#!/usr/bin/env python3
"""Fine-tune DocDiff on these books: scan -> scanclean output.

    python train_docdiff.py OUTDIR [--iters N]

Starts from the authors' deblurring weights. Loss is the paper's: coarse
predictor pixel loss (+ low-frequency term) and x0-prediction diffusion loss on
the residual (+ Laplacian high-frequency term), beta = 50, T = 100.
Held out (never seen in training): HOLD pages below.
"""
import argparse, os, sys, time, random
import cv2, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_docdiff import DD, load, dev
sys.path.insert(0, DD)
from schedule.schedule import Schedule
from src.sobel import Laplacian

HOLD = {"s2p03", "s2p04", "s1p06", "s4p03", "s3p04"}
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--size", type=int, default=128)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

tags = sorted(f[:-4] for f in os.listdir(f"{HERE}/in") if f.endswith(".png") and f[:-4] not in HOLD)
X = [cv2.imread(f"{HERE}/in/{t}.png")[:, :, ::-1].copy() for t in tags]
Y = [cv2.imread(f"{HERE}/sc/{t}.png", 0) for t in tags]
# sample where there is something to learn: ink in the input or a removal
wts = []
for x, y in zip(X, Y):
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    interest = ((g < 150) | (np.abs(g.astype(int) - y) > 60)).astype(np.float32)
    wts.append(cv2.boxFilter(interest, -1, (a.size, a.size)))


def batch():
    xs, ys = [], []
    S = a.size
    for _ in range(a.batch):
        k = random.randrange(len(X))
        H, W = Y[k].shape
        for _ in range(20):
            i, j = random.randrange(H - S), random.randrange(W - S)
            if wts[k][i + S // 2, j + S // 2] > 0.02 or random.random() < 0.15:
                break
        xs.append(X[k][i:i + S, j:j + S]); ys.append(Y[k][i:i + S, j:j + S])
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(np.stack(ys))[:, None].float().repeat(1, 3, 1, 1) / 255
    return x.to(dev), y.to(dev)


net = load("deblur").train()
T = 100
betas = Schedule("linear", T).get_betas().to(dev).float()
g = torch.cumprod(1 - betas, 0)
sg, s1g = g.sqrt(), (1 - g).sqrt()
lap = Laplacian().to(dev)
mse = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
t0 = time.time()
for it in range(1, a.iters + 1):
    x, y = batch()
    init = net.init_predictor(x, 0)
    res = y - init
    t = torch.randint(0, T, (x.shape[0],), device=dev)
    noise = torch.randn_like(res)
    noisy = sg[t][:, None, None, None] * res + s1g[t][:, None, None, None] * noise
    pred = net.denoiser(torch.cat((noisy, init.detach() if False else init), 1), t)
    ddpm = 2 * mse(lap(pred), lap(res)) + mse(pred, res)
    pix = mse(init, y) + 2 * mse(init - lap(init), y - lap(y))
    loss = ddpm + 50 * pix / T
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0:
        print(f"it {it} loss {loss.item():.4f} ddpm {ddpm.item():.4f} pix {pix.item():.4f} "
              f"{(time.time()-t0)/it:.2f}s/it", flush=True)
    if it % 2000 == 0 or it == a.iters:
        torch.save(net.init_predictor.state_dict(), f"{a.out}/init.pth")
        torch.save(net.denoiser.state_dict(), f"{a.out}/denoiser.pth")
print("done", time.time() - t0)
