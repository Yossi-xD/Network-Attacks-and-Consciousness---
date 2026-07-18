"""Build the final labelled 'Subject A | Morph (alpha=0.5) | Subject B'
figures for the pairs chosen to showcase in the report/submission."""
import os
import sys
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from img_io import imread, imwrite

# Keep generated assets inside the project, regardless of where the script is run.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PAIRS_DIR = os.path.join(BASE, "outputs", "pairs")
MORPHS_DIR = os.path.join(BASE, "outputs", "morphs")
SHOWCASE_DIR = os.path.join(BASE, "outputs", "showcase")
os.makedirs(SHOWCASE_DIR, exist_ok=True)

CHOSEN = ["pair000", "pair003", "pair004", "pair007", "pair009"]

LABEL_H = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX


def labeled(img, text):
    canvas = np.full((img.shape[0] + LABEL_H, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[LABEL_H:] = img
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    x = max(2, (img.shape[1] - tw) // 2)
    cv2.putText(canvas, text, (x, 20), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


for pid in CHOSEN:
    imgA = imread(os.path.join(PAIRS_DIR, f"{pid}_A.png"))
    imgB = imread(os.path.join(PAIRS_DIR, f"{pid}_B.png"))
    morphed = imread(os.path.join(MORPHS_DIR, f"{pid}_morph.png"))

    a = labeled(imgA, "Subject A (bona fide)")
    m = labeled(morphed, "Morph (alpha=0.5)")
    b = labeled(imgB, "Subject B (bona fide)")

    sep = np.full((a.shape[0], 6, 3), 200, dtype=np.uint8)
    triptych = np.hstack([a, sep, m, sep, b])
    imwrite(os.path.join(SHOWCASE_DIR, f"{pid}_triptych.png"), triptych)

print(f"Wrote {len(CHOSEN)} showcase triptychs to {SHOWCASE_DIR}")
