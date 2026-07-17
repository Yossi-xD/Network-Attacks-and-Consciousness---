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

Reimplements the single-image morphing attack detection (S-MAD) method of
Venkatesh, Raghavendra, Raja & Busch, "Single Image Face Morphing Attack
Detection Using Ensemble of Features" (IEEE FUSION 2020), trained and
evaluated on the Part 1 bona fide/morph dataset.

Run the pipeline:

```powershell
python src/train_mad.py            # trains + evaluates, writes outputs/mad/
python src/train_mad_nn.py          # trains the neural fusion layer on top
python src/build_mad_report_pdf.py  # renders submission_report_part2.pdf
```

Pipeline: YCbCr + HSV color-space expansion -> 3-level Laplacian pyramid ->
LBP / HOG / BSIF descriptors per scale-space image -> a Collaborative
Representation Classifier per descriptor stream -> sum-rule score fusion.
Evaluated with 5-fold pair-disjoint cross-validation and the ISO/IEC 30107-3
metrics (APCER, BPCER, D-EER).

On top of that, `src/mad_nn.py` + `src/train_mad_nn.py` replace the fixed
sum-rule fusion with a trained single-neuron logistic regression (the
simplest possible neural network) over the same 3 out-of-fold stream
scores: full-batch gradient descent on binary cross-entropy, reporting
training/test accuracy, the loss curve, and the learned weights and bias
per stream. See `submission_report_part2.pdf` for the full method,
adaptations from the paper, results, and limitations.
