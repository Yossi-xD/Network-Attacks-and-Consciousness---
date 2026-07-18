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

## Part 2: Morphing Attack Detection

Part 2 lives in [Part2_Submission/](Part2_Submission/): an implementation of
**DMorphNet** (Gawade et al., COMPUTATIA 2026,
[DOI 10.2991/978-94-6239-713-2_20](https://doi.org/10.2991/978-94-6239-713-2_20))
— EfficientNet-B6 deep features + an SVM classifier for real-vs-morph
detection. See that folder's README for the full citation, run instructions
(local PC or Colab), and results.

## Submission packages

- `Part1_Submission/` + `Part1_Submission.zip` — morphing (report, pairs,
  morphs, code)
- `Part2_Submission/` + `Part2_Submission.zip` — detection (article,
  notebook, analysis report, presentation)
