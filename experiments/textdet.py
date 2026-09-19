"""PP-OCRv6 tiny text detector (Apache-2.0, 1.8 MB ONNX) via OpenCV DNN.

Reproduces PaddleOCR's DB post-processing: probability > 0.2, keep regions
whose mean probability > 0.4, grow each by area*1.4/perimeter (the 'unclip'),
as a filled min-area rectangle. Returns (mask uint8, boxes [(x,y,w,h,poly)]).
"""
import os
import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "ppocrv6_tiny_det.onnx")
_net = None


def detect(bgr, thresh=0.2, box_thresh=0.4, unclip=1.4, max_side=960):
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(MODEL)
    H, W = bgr.shape[:2]
    r = min(1.0, max_side / max(H, W)) if max_side else 1.0
    h, w = max(32, int(round(H * r / 32)) * 32), max(32, int(round(W * r / 32)) * 32)
    x = cv2.resize(bgr, (w, h)).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    _net.setInput(x.transpose(2, 0, 1)[None])
    prob = _net.forward()[0, 0]
    prob = cv2.resize(prob, (W, H))
    bit = (prob > thresh).astype(np.uint8)
    cs, _ = cv2.findContours(bit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((H, W), np.uint8)
    boxes = []
    for c in cs[:3000]:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 3:
            continue
        # score = mean probability inside the min-area box, not the contour:
        # a ragged edge strip scores high along its contour but low over its box
        pts = cv2.boxPoints(rect)
        x0, y0 = np.floor(pts.min(0)).astype(int).clip(0)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x1, y1 = min(x1, W - 1), min(y1, H - 1)
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [(pts - (x0, y0)).astype(np.int32)], 1)
        if cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], m)[0] < box_thresh:
            continue
        d = rw * rh * unclip / (2 * (rw + rh))          # offset of the box, as pyclipper does
        if min(rw, rh) + 2 * d < 5:
            continue
        poly = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), ang)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        bx, by, bw2, bh2 = cv2.boundingRect(poly)
        boxes.append((bx, by, bw2, bh2, poly))
    detect.prob = prob
    return mask, boxes
