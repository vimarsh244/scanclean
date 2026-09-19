#!/usr/bin/env python3
"""SauvolaNet (ICDAR 2021, MIT) - pretrained weights, ported from Keras to torch.

    python run_sauvolanet.py OUTDIR [src=in|norm] tags|all

Multi-window Sauvola thresholds (w = 7..63, learned k, R) blended per pixel by a
6-layer dilated-conv attention branch; text where (img - T) * alpha < 0.
Input normalised as in the authors' DataGenerator: (x - min) / (max - min + .1).
"""
import os, sys, json, time
import cv2, h5py, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(os.path.dirname(HERE), "third_party", "sauvolanet", "sauvola.h5")
f = h5py.File(H5, "r")
W = f["model_weights"]
cfg = json.loads(f.attrs["model_config"])
wins = [l for l in cfg["config"]["layers"] if l["name"] == "sauvola"][0]["config"]["window_size_list"]
k = torch.tensor(np.array(W["sauvola/sauvola/Sauvola_k:0"]).reshape(-1))
R = torch.tensor(np.array(W["sauvola/sauvola/Sauvola_R:0"]).reshape(-1))
alpha = float(np.array(W["difference_thresh/difference_thresh/alpha:0"]).ravel()[0])
convs = []
for i in range(6):
    kk = torch.tensor(np.array(W[f"conv{i}/conv{i}/kernel:0"])).permute(3, 2, 0, 1)
    bb = torch.tensor(np.array(W[f"conv{i}/conv{i}/bias:0"]))
    convs.append((kk, bb, 1 if i == 0 else 2))
att_k = torch.tensor(np.array(W["conv_att/conv_att/kernel:0"])).permute(3, 2, 0, 1)
att_b = torch.tensor(np.array(W["conv_att/conv_att/bias:0"]))
print("windows", wins, "k", k.numpy().round(3), "R", R.numpy().round(3), "alpha", round(alpha, 2))


def box_mean(x, w):
    s = cv2.boxFilter(x, cv2.CV_64F, (w, w), normalize=False, borderType=cv2.BORDER_CONSTANT)
    c = cv2.boxFilter(np.ones_like(x), cv2.CV_64F, (w, w), normalize=False, borderType=cv2.BORDER_CONSTANT)
    return s / c


@torch.no_grad()
def run(gray):
    x = gray.astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 0.1)
    Ts = []
    for i, w in enumerate(wins):
        m, m2 = box_mean(x, w), box_mean(x * x, w)
        sd = np.sqrt(np.maximum(m2 - m * m, 1e-6))
        Ts.append(m * (1 + float(k[i]) * (sd / float(R[i]) - 1)))
    T = torch.tensor(np.stack(Ts), dtype=torch.float32)            # n,H,W
    t = torch.tensor(x, dtype=torch.float32)[None, None]
    h = t
    for kk, bb, d in convs:
        h = F.conv2d(h, kk, bb, padding=d, dilation=d)
        mu = h.mean(dim=(2, 3), keepdim=True)
        sd = h.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        h = F.relu((h - mu) / sd)
    a = torch.softmax(F.conv2d(h, att_k, att_b, padding=1), dim=1)[0]  # n,H,W
    th = (a * T).sum(0)
    diff = (t[0, 0] - th) * alpha
    return diff.numpy()


if __name__ == "__main__":
    out = sys.argv[1]; src = sys.argv[2]; tags = sys.argv[3:]
    if tags == ["all"]:
        tags = sorted(f[:-4] for f in os.listdir(f"{HERE}/in") if f.endswith(".png"))
    os.makedirs(out, exist_ok=True)
    for tg in tags:
        g = cv2.imread(f"{HERE}/{src}/{tg}.png", 0)
        t0 = time.time()
        d = run(g)
        cv2.imwrite(f"{out}/{tg}.png", np.where(d < 0, 0, 255).astype(np.uint8))
        print(tg, f"{time.time()-t0:.1f}s", flush=True)
