# models

`ppocrv6_tiny_det.onnx` — PP-OCRv6 tiny text detector (PaddleOCR, Apache-2.0,
0.43M params), converted from the official inference model with
`paddle2onnx --opset_version 13`. Loaded by `scanclean.detect_text()` through
OpenCV 5's DNN module; used only to locate lines of type, never to paint pixels.
Source: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html
