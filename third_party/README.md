# third_party

Upstream code and weights used only by `experiments/` (never by the packaged
ScanClean pipeline, whose one learned component is the bundled
`models/ppocrv6_tiny_det.onnx`).

| Dir | Upstream | License | What is here |
| --- | --- | --- | --- |
| `docdiff/` | [Royalvice/DocDiff](https://github.com/Royalvice/DocDiff) (ACM MM 2023) | MIT | `model/DocDiff.py`, `schedule/schedule.py`, `src/sobel.py`, `fetch_weights.py` |
| `sauvolanet/` | [Leedeng/SauvolaNet](https://github.com/Leedeng/SauvolaNet) (ICDAR 2021) | MIT | pretrained `sauvola.h5` (the Keras model is re-implemented in `experiments/run_sauvolanet.py`) |
| `zigzag/` | [Bloechle/ZigZag](https://github.com/Bloechle/ZigZag) (DocEng 2024) | AGPL-3.0 | the authors' Python port, `zigzag.py`, unmodified |

Local changes: `docdiff/model/DocDiff.py` — the `UNet.forward` default argument
`torch.tensor([0]).cuda()` became `torch.tensor([0])`, so it imports without CUDA.

DocDiff's weights (64 MB) are not committed:
`python3 third_party/docdiff/fetch_weights.py` puts them in `docdiff/checksave/`.
