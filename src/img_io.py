"""Unicode-path-safe image I/O.

cv2.imread/imwrite call the C runtime's fopen() with a narrow (ANSI) path on
Windows, so any path containing non-ASCII characters (this project lives
under a Hebrew course-name folder) silently fails. Routing through
numpy's own file I/O plus cv2.imencode/imdecode sidesteps that entirely.
"""
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"Failed to decode image: {path}")
    return img


def imwrite(path, img):
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Failed to encode image for: {path}")
    buf.tofile(path)
