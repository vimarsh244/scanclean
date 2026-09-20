import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const [bundleDirectory, pyodidePackageDirectory] = process.argv.slice(2);
if (!bundleDirectory || !pyodidePackageDirectory) {
  throw new Error("usage: node test_pyodide_bundle.mjs BUNDLE_DIR PYODIDE_PACKAGE_DIR");
}

const moduleUrl = pathToFileURL(
  path.join(pyodidePackageDirectory, "node_modules", "pyodide", "pyodide.mjs"),
).href;
const { loadPyodide } = await import(moduleUrl);
// Pyodide's Node loader resolves this with node:path; passing a file: URL makes
// it prepend the working directory on Windows. An absolute path works in Node
// on every CI host and still provides the base for relative wheel paths.
const lockFileURL = path.resolve(bundleDirectory, "pyodide-lock.json");
const pyodide = await loadPyodide({
  lockFileURL,
  // Node resolves locked packages from this cache directory. Browsers derive
  // the equivalent base URL from lockFileURL.
  packageCacheDir: path.resolve(bundleDirectory),
});
await pyodide.loadPackage("scanclean");

const result = await pyodide.runPythonAsync(`
import cv2
import numpy as np
from scanclean import Options, clean_page
from scanclean.core import MODEL

assert cv2.__version__ == "5.0.0"
net = cv2.dnn.readNetFromONNX(str(MODEL))
net.setInput(np.zeros((1, 3, 320, 320), dtype=np.float32))
prediction = net.forward()
assert prediction.ndim == 4

page = np.full((64, 96, 4), 255, dtype=np.uint8)
page[20:44, 24:72, :3] = 32
bgr = cv2.cvtColor(page, cv2.COLOR_RGBA2BGR)
cleaned, stats, audit, dpi = clean_page(
    bgr, dpi=300, opts=Options(detect=False, deskew=False)
)
assert cleaned.dtype == np.uint8
assert cleaned.ndim == 2
assert dpi == 300
True
`);
assert.equal(result, true);
console.log("Pyodide bundle smoke test passed with OpenCV 5.0.0");
