"""Interactive face-morphing demo app (Streamlit).

Upload two face photos, pick alpha, and morph them using the exact same
landmark/triangulation/warp pipeline in src/landmarks.py + src/morph.py.

Must be launched with the process working directory set to the ASCII
junction (C:\\Users\\Yossi\\face_morph_project_run) -- see README's Unicode/Windows
note; run_app.ps1 in the project root handles this.
"""
import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import get_landmarks
from morph import delaunay_triangulation, morph_images, _warp_triangle

st.set_page_config(page_title="Face Morph Studio", page_icon="\U0001F9EC", layout="wide")

st.markdown(
    """
    <style>
      h1, h2, h3 { font-family: "Bahnschrift", "Segoe UI Semibold", sans-serif; }
      .stCaption, code { font-family: "Cascadia Mono", "Consolas", monospace; }
      div[data-testid="stFileUploader"] section { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SQUARE_SIZE = 480
TARGET_MARGIN = 2.3    # aspirational crop side = this * face bbox size
DETECT_MAX_DIM = 1200  # only downscales the *bbox-finding* pass, not the final crop
WORK_MAX_DIM = 1600    # caps memory/compute for the original decoded photo


def _clamp_square(cx, cy, side, w, h):
    side = min(side, w, h)
    half = side / 2
    x0 = min(max(cx - half, 0), w - side)
    y0 = min(max(cy - half, 0), h - side)
    return int(round(x0)), int(round(y0)), int(round(side))


@st.cache_data(show_spinner=False)
def _face_bbox(file_bytes):
    """Decode + a first detection pass to locate the face. Returns
    (img, cx, cy, bbox_size) in the (possibly downscaled) working image's own
    pixel coordinates, or None if no face was found."""
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > WORK_MAX_DIM:
        s = WORK_MAX_DIM / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    probe = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    pts, _ = get_landmarks(probe, upscale=1)
    if pts is None:
        return None
    face_pts = pts[:478] / scale  # exclude the 8 synthetic boundary points
    (x0, y0), (x1, y1) = face_pts.min(axis=0), face_pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return img, cx, cy, max(x1 - x0, y1 - y0)


@st.cache_data(show_spinner=False)
def prepare_pair(file_bytes_a, file_bytes_b, target_margin=TARGET_MARGIN, size=SQUARE_SIZE):
    """Crop both photos to a *shared* face-to-frame ratio before landmarking.

    Cropping each photo to its own independently-clamped margin isn't enough:
    most ordinary headshots already fill 40-60%+ of the frame, so an
    independent per-photo clamp silently falls back to a plain center-crop
    for BOTH photos in the common case, reproducing the original mismatch.
    Instead we find the largest margin that's simultaneously achievable for
    *both* photos and apply that same margin to each, so the two faces
    always end up at a matching scale -- which is what the triangulated warp
    actually needs to avoid smearing hair/background into a double-exposure.
    """
    info_a = _face_bbox(file_bytes_a)
    info_b = _face_bbox(file_bytes_b)
    if info_a is None or info_b is None:
        return None, None, None, None

    img_a, cxa, cya, ba = info_a
    img_b, cxb, cyb, bb = info_b
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    feasible_a = min(wa, ha) / ba
    feasible_b = min(wb, hb) / bb
    margin = min(target_margin, feasible_a, feasible_b)

    xa0, ya0, sa = _clamp_square(cxa, cya, ba * margin, wa, ha)
    xb0, yb0, sb = _clamp_square(cxb, cyb, bb * margin, wb, hb)
    crop_a = cv2.resize(img_a[ya0:ya0 + sa, xa0:xa0 + sa], (size, size), interpolation=cv2.INTER_AREA)
    crop_b = cv2.resize(img_b[yb0:yb0 + sb, xb0:xb0 + sb], (size, size), interpolation=cv2.INTER_AREA)

    pts_a, up_a = get_landmarks(crop_a, upscale=1)
    pts_b, up_b = get_landmarks(crop_b, upscale=1)
    return pts_a, up_a, pts_b, up_b


def _warp_to_shape(img, pts_src, pts_dst, triangles):
    """Warp img from its own landmark shape to a target shape, reusing
    morph.py's per-triangle affine warp (the graded code, not a rewrite)."""
    imgf = img.astype(np.float32)
    out = np.zeros_like(imgf)
    for (i, j, k) in triangles:
        _warp_triangle(
            imgf, out,
            [tuple(pts_src[i]), tuple(pts_src[j]), tuple(pts_src[k])],
            [tuple(pts_dst[i]), tuple(pts_dst[j]), tuple(pts_dst[k])],
        )
    return out


def render_morph(up_a, pts_a, up_b, pts_b, alpha, frame_from, triangles=None):
    """frame_from=None -> the report's raw full-frame cross-dissolve
    (morph_images verbatim). frame_from='A'/'B' -> blend only the face region
    and take hair + background from that subject's warped photo.

    Hair and background have no landmarks, so a full-frame cross-dissolve can
    only ghost them over each other (e.g. long hair fading over short hair).
    Compositing the blended face into one contributor's photo is how morphing
    attacks are actually assembled: the photo passes as one person's, with the
    blended identity confined to the face.
    """
    if frame_from is None:
        return morph_images(up_a, pts_a, up_b, pts_b, alpha, triangles=triangles)

    h, w = up_a.shape[:2]
    pts_m = (1 - alpha) * pts_a + alpha * pts_b
    if triangles is None:
        triangles = delaunay_triangulation(pts_m, (w, h))
    warped_a = _warp_to_shape(up_a, pts_a, pts_m, triangles)
    warped_b = _warp_to_shape(up_b, pts_b, pts_m, triangles)
    blend = (1 - alpha) * warped_a + alpha * warped_b
    frame = warped_a if frame_from == "A" else warped_b

    hull = cv2.convexHull(pts_m[:478].astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    blend8 = np.clip(blend, 0, 255).astype(np.uint8)
    frame8 = np.clip(frame, 0, 255).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(hull)
    try:
        # Poisson blending harmonizes the pasted face's skin tone/lighting
        # with the frame photo at the seam.
        out = cv2.seamlessClone(blend8, frame8, mask,
                                (x + bw // 2, y + bh // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone rejects hulls that touch the image border (very
        # tightly-cropped sources) -- fall back to a feathered alpha paste.
        soft = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), np.uint8)), (31, 31), 0)
        m3 = (soft.astype(np.float32) / 255.0)[..., None]
        out = np.clip(blend * m3 + frame * (1 - m3), 0, 255).astype(np.uint8)
    return out, triangles


st.title("Face Morph Studio")
st.caption(
    "Upload two face photos and blend them with the landmark-morphing pipeline "
    "from Part 1 of the report — MediaPipe 478-pt mesh → Delaunay triangulation "
    "→ per-triangle affine warp → cross-dissolve."
)

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Subject A", type=["jpg", "jpeg", "png", "bmp"], key="file_a")
with col_b:
    file_b = st.file_uploader("Subject B", type=["jpg", "jpeg", "png", "bmp"], key="file_b")

alpha = st.slider("Alpha (blend factor)", 0.0, 1.0, 0.5, 0.05)
frame_choice = st.radio(
    "Hair & background",
    ["Subject A's photo", "Subject B's photo", "Blend both (raw cross-dissolve)"],
    horizontal=True,
    help="Hair and background have no landmarks, so blending them can only "
         "ghost the two photos over each other. Real morphing attacks blend "
         "the face only and take hair/background from one subject's photo.",
)
frame_from = {"Subject A's photo": "A", "Subject B's photo": "B"}.get(frame_choice)

if file_a and file_b:
    pts_a, up_a, pts_b, up_b = prepare_pair(file_a.getvalue(), file_b.getvalue())

    if pts_a is None:
        st.error("Couldn't detect a face in Subject A's photo — try a clearer, front-facing shot.")
    elif pts_b is None:
        st.error("Couldn't detect a face in Subject B's photo — try a clearer, front-facing shot.")
    else:
        morphed, _ = render_morph(up_a, pts_a, up_b, pts_b, alpha, frame_from)

        c1, c2, c3 = st.columns(3)
        c1.image(up_a, channels="BGR", caption="Subject A", use_container_width=True)
        c2.image(morphed, channels="BGR", caption=f"Morph — alpha={alpha:.2f}", use_container_width=True)
        c3.image(up_b, channels="BGR", caption="Subject B", use_container_width=True)

        ok, buf = cv2.imencode(".png", morphed)
        if ok:
            st.download_button(
                "Download morph (PNG)", data=buf.tobytes(),
                file_name=f"morph_alpha{alpha:.2f}.png", mime="image/png",
            )

        with st.expander("Show 5-step interpolation strip (alpha = 0 / 0.25 / 0.5 / 0.75 / 1)"):
            if st.button("Generate strip"):
                w, h = SQUARE_SIZE, SQUARE_SIZE
                pts_mid = (1 - 0.5) * pts_a + 0.5 * pts_b
                triangles = delaunay_triangulation(pts_mid, (w, h))
                tiles = []
                for a in (0.0, 0.25, 0.5, 0.75, 1.0):
                    im, _ = render_morph(up_a, pts_a, up_b, pts_b, a, frame_from,
                                         triangles=triangles)
                    tiles.append(im)
                strip = np.hstack(tiles)
                st.image(strip, channels="BGR", use_container_width=True)
else:
    st.info("Upload two front-facing photos above to generate a morph.")
