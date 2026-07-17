# Face Morphing Final Project

This project creates face morphs from two facial images using a custom
landmark-based OpenCV pipeline. It satisfies the assignment requirement of
at least five original-face pairs and their morphed results.

## Submission files

This project provides an interactive face-morphing dashboard. Upload two face
images to create a landmark-based morph in the browser.

## Run the app

From PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Open <http://localhost:8501>, upload two face images, and move the alpha
slider to create a morph. The launcher creates an ASCII-only Windows junction
because MediaPipe cannot start from this folder's Hebrew path.

## Method

1. MediaPipe detects 478 corresponding facial landmarks in both images.
2. Eight image-boundary points are added to keep the full frame aligned.
3. Delaunay triangulation is computed on the alpha-averaged landmark shape.
4. Each triangle is affine-warped from both source images to that target shape.
5. The warped images are linearly cross-dissolved using alpha.

