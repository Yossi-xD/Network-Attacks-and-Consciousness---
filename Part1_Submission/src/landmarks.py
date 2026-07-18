"""Facial landmark detection wrapper around MediaPipe FaceLandmarker."""
import os
import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python

# The launcher changes the working directory to an ASCII-only junction before
# importing this module.  MediaPipe's native component requires that on Windows
# when the real project path contains Hebrew characters.
MODEL_PATH = os.path.join(os.getcwd(), "models", "face_landmarker.task")

_options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
)
_landmarker = vision.FaceLandmarker.create_from_options(_options)

# 8 fixed boundary points (image corners + edge midpoints) so the whole
# frame -- not just the face -- participates in the triangulation/warp.
def _boundary_points(w, h):
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2], [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
    ], dtype=np.float64)


def get_landmarks(img_bgr, upscale=3):
    """Return Nx2 float array of landmark pixel coords (face mesh + 8 frame
    boundary points), in the coordinate system of the *upscaled* image, plus
    the upscaled BGR image itself. Returns (None, None) if no face found."""
    img_bgr = cv2.resize(img_bgr, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
                    dtype=np.float64)
    pts = np.vstack([pts, _boundary_points(w, h)])
    return pts, img_bgr
