"""Fetch LFW (Labeled Faces in the Wild) face-pair data for the morphing project.

Uses scikit-learn's built-in LFW pairs loader, which ships pre-built pairs of
"different person" (target=0) and "same person" (target=1) images. We only
need different-person pairs, since face-morphing attacks combine two distinct
identities into one image.
"""
import os
import sys
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from img_io import imwrite

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    print("Downloading/loading LFW pairs (this can take a while the first time)...")
    lfw = fetch_lfw_pairs(subset="train", color=True, resize=1.0, funneled=True,
                           data_home=os.path.join(DATA_DIR, "sklearn_cache"))
    pairs = lfw.pairs  # shape (N, 2, H, W, 3), float in [0,1]
    targets = lfw.target  # 1 = same person, 0 = different person
    target_names = lfw.target_names
    print("pairs shape:", pairs.shape, "targets:", np.bincount(targets))
    print("target_names:", target_names)

    diff_idx = np.where(targets == 0)[0]
    print(f"Found {len(diff_idx)} different-person pairs")

    out_dir = os.path.join(DATA_DIR, "lfw_diff_pairs")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    chosen = rng.choice(diff_idx, size=min(60, len(diff_idx)), replace=False)

    for rank, idx in enumerate(chosen):
        imgA = (pairs[idx, 0] * 255).astype(np.uint8)
        imgB = (pairs[idx, 1] * 255).astype(np.uint8)
        imgA_bgr = cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR)
        imgB_bgr = cv2.cvtColor(imgB, cv2.COLOR_RGB2BGR)
        imwrite(os.path.join(out_dir, f"pair{rank:03d}_A.png"), imgA_bgr)
        imwrite(os.path.join(out_dir, f"pair{rank:03d}_B.png"), imgB_bgr)

    print(f"Saved {len(chosen)} candidate pairs to {out_dir}")


if __name__ == "__main__":
    main()
