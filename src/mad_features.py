"""Feature extraction for single-image morphing-attack detection (S-MAD),
following Venkatesh, Raghavendra, Raja & Busch, "Single Image Face Morphing
Attack Detection Using Ensemble of Features" (IEEE FUSION 2020): color-space
expansion (YCbCr + HSV) -> Laplacian-pyramid scale space -> LBP / HOG / BSIF
texture descriptors, one long feature vector per descriptor per image.

Deviation from the paper: BSIF normally uses a filter bank pre-learned via
ICA on a large natural-image corpus (Kannala & Rahtu, 2012). We do not have
that corpus, so `learn_bsif_filters` learns an equivalent small ICA filter
bank directly from the bona fide training images of each cross-validation
fold instead -- same idea (statistically independent texture filters),
different (and fold-local, leakage-free) training data.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import FastICA

RESIZE = (96, 96)
PYRAMID_LEVELS = 3
LBP_RADIUS, LBP_POINTS = 1, 8
BLOCK, STRIDE = 20, 10
BSIF_PATCH = 5
BSIF_N_FILTERS = 6  # -> 2**6 = 64-bin BSIF code


def to_color_channels(img_bgr):
    """The paper's 'Ic_i, i=1..6': the 3 channels of YCbCr and of HSV."""
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return [ycc[:, :, i] for i in range(3)] + [hsv[:, :, i] for i in range(3)]


def laplacian_pyramid(gray, levels=PYRAMID_LEVELS):
    """Classic Laplacian pyramid: `levels-1` high-frequency detail bands plus
    one coarse Gaussian residual, capturing morphing-warp residue at
    multiple scales."""
    g = [gray.astype(np.float32)]
    for _ in range(levels - 1):
        g.append(cv2.pyrDown(g[-1]))
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _normalize_u8(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def _resize(img):
    return cv2.resize(_normalize_u8(img), RESIZE, interpolation=cv2.INTER_AREA)


def scale_space_images(img_bgr):
    """The 18 resized scale-space sub-images: 6 color channels x 3
    Laplacian-pyramid levels."""
    out = []
    for ch in to_color_channels(img_bgr):
        for level in laplacian_pyramid(ch):
            out.append(_resize(level))
    return out


def _block_histogram(code_img, n_bins):
    h, w = code_img.shape
    feats = []
    for y in range(0, h - BLOCK + 1, STRIDE):
        for x in range(0, w - BLOCK + 1, STRIDE):
            block = code_img[y:y + BLOCK, x:x + BLOCK]
            hist, _ = np.histogram(block, bins=n_bins, range=(0, n_bins))
            feats.append(hist.astype(np.float32) / (block.size + 1e-6))
    return np.concatenate(feats)


def lbp_descriptor(gray_u8):
    lbp = local_binary_pattern(gray_u8, LBP_POINTS, LBP_RADIUS, method="uniform")
    return _block_histogram(lbp.astype(np.int32), LBP_POINTS + 2)


def hog_descriptor(gray_u8):
    return hog(gray_u8, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)


def learn_bsif_filters(sample_images, patch_size=BSIF_PATCH,
                        n_filters=BSIF_N_FILTERS, n_patches=4000, seed=0):
    """Learn an ICA filter bank from random patches of `sample_images`
    (grayscale uint8), approximating BSIF's statistically-independent
    texture filters (see module docstring for the deviation from the
    original natural-image-trained bank)."""
    rng = np.random.default_rng(seed)
    per_image = max(1, n_patches // max(len(sample_images), 1) + 1)
    patches = []
    for img in sample_images:
        h, w = img.shape
        if h <= patch_size or w <= patch_size:
            continue
        for _ in range(per_image):
            y = rng.integers(0, h - patch_size)
            x = rng.integers(0, w - patch_size)
            patches.append(img[y:y + patch_size, x:x + patch_size].astype(np.float32).ravel())
    X = np.array(patches)
    X -= X.mean(axis=1, keepdims=True)
    ica = FastICA(n_components=n_filters, random_state=seed, max_iter=1000, whiten="unit-variance")
    ica.fit(X)
    return ica.components_.reshape(n_filters, patch_size, patch_size).astype(np.float32)


def bsif_descriptor(gray_u8, filters):
    gray_f = gray_u8.astype(np.float32)
    code = np.zeros(gray_u8.shape, dtype=np.int32)
    for i, f in enumerate(filters):
        resp = cv2.filter2D(gray_f, -1, f, borderType=cv2.BORDER_REFLECT_101)
        code += ((resp > 0).astype(np.int32) << i)
    return _block_histogram(code, 2 ** len(filters))


def lbp_stream(sub_images):
    return np.concatenate([lbp_descriptor(g) for g in sub_images])


def hog_stream(sub_images):
    return np.concatenate([hog_descriptor(g) for g in sub_images])


def bsif_stream(sub_images, filters):
    return np.concatenate([bsif_descriptor(g, filters) for g in sub_images])
