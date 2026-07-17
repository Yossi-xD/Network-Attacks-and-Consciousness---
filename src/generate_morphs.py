"""Generate morphs for all fetched LFW different-person pairs.

For every pair (A, B):
  - detect landmarks on both faces
  - build a Delaunay triangulation on the averaged (alpha=0.5) landmarks
  - render the morph at alpha=0.5 (the standard "attack" morph) plus a
    0/0.25/0.5/0.75/1 interpolation strip for a subset used in the report
Skips any pair where landmark detection fails on either image.
"""
import os
import sys
import json
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from img_io import imread, imwrite
from landmarks import get_landmarks
from morph import morph_images

# Resolve from this file rather than an old, machine-specific junction name.
# This works both from the repository and from run_app.ps1's ASCII junction.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PAIRS_DIR = os.path.join(BASE, "data", "lfw_diff_pairs")
OUT_PAIRS = os.path.join(BASE, "outputs", "pairs")
OUT_MORPHS = os.path.join(BASE, "outputs", "morphs")
OUT_STRIPS = os.path.join(BASE, "outputs", "strips")
os.makedirs(OUT_PAIRS, exist_ok=True)
os.makedirs(OUT_MORPHS, exist_ok=True)
os.makedirs(OUT_STRIPS, exist_ok=True)

N_PAIRS_TO_TRY = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAKE_STRIPS_FOR = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # how many get the alpha strip

pair_ids = sorted({f.split("_")[0] for f in os.listdir(PAIRS_DIR)})[:N_PAIRS_TO_TRY]

manifest = []
n_ok, n_fail = 0, 0
for k, pid in enumerate(pair_ids):
    pathA = os.path.join(PAIRS_DIR, f"{pid}_A.png")
    pathB = os.path.join(PAIRS_DIR, f"{pid}_B.png")
    imgA = imread(pathA)
    imgB = imread(pathB)

    ptsA, upA = get_landmarks(imgA)
    ptsB, upB = get_landmarks(imgB)
    if ptsA is None or ptsB is None:
        n_fail += 1
        continue

    morphed, triangles = morph_images(upA, ptsA, upB, ptsB, alpha=0.5)

    imwrite(os.path.join(OUT_PAIRS, f"{pid}_A.png"), upA)
    imwrite(os.path.join(OUT_PAIRS, f"{pid}_B.png"), upB)
    imwrite(os.path.join(OUT_MORPHS, f"{pid}_morph.png"), morphed)

    if k < MAKE_STRIPS_FOR:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        tiles = []
        for a in alphas:
            im, _ = morph_images(upA, ptsA, upB, ptsB, alpha=a, triangles=triangles)
            tiles.append(im)
        strip = np.hstack(tiles)
        imwrite(os.path.join(OUT_STRIPS, f"{pid}_strip.png"), strip)

    manifest.append(pid)
    n_ok += 1

print(f"ok={n_ok} fail={n_fail} total_tried={len(pair_ids)}")
with open(os.path.join(BASE, "outputs", "morph_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
