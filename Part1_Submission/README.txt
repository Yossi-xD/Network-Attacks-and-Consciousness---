Face Morphing - Part 1 Submission
==================================

WHERE EACH ASSIGNMENT REQUIREMENT IS (A-F):

A. At least 5 pairs of original faces + their morphed results
   -> results/showcase/       : the 5 triptychs (Subject A | Morph | Subject B) shown in the report
   -> results/strips/         : 10 side-by-side strips (originals + morph in one image)
   -> results/pairs/          : ALL 58 original bona fide pairs (pairXXX_A.png, pairXXX_B.png)
   -> results/morphs/         : ALL 58 morphed results (pairXXX_morph.png), alpha = 0.5
   (Requirement asks for at least 5 pairs -- we submit 58.)

B. Short report (max 2 pages)
   -> submission_report.pdf   : exactly 2 pages, contains sections C-F below

C. Method used
   -> submission_report.pdf, Section 1 ("Method Used")
      Landmark-based morphing: MediaPipe landmarks -> Delaunay triangulation
      -> piecewise-affine warping -> cross-dissolve. Code: src/morph.py

D. Landmark alignment strategy
   -> submission_report.pdf, Section 2 ("Landmark Alignment Strategy")
      478 MediaPipe FaceLandmarker points + 8 frame-boundary points,
      index-based correspondence. Code: src/landmarks.py

E. Interpolation technique
   -> submission_report.pdf, Section 3 ("Interpolation Technique")
      Geometric: per-triangle affine warp (cv2.getAffineTransform / warpAffine).
      Photometric: linear cross-dissolve with weights (1-alpha, alpha).
      Code: src/morph.py (_warp_triangle, morph_images)

F. Observations and limitations
   -> submission_report.pdf, Section 4 ("Observations and Limitations")

SOURCE CODE (src/):
   morph.py            - Delaunay triangulation + piecewise-affine warp + blend
   landmarks.py        - MediaPipe landmark detection + boundary points
   img_io.py           - image loading helpers
   app.py              - interactive Streamlit demo (upload 2 faces, alpha slider)
   build_showcase.py   - renders the triptych/strip images from pairs+morphs
   build_report_pdf.py - renders submission_report.pdf

RUN THE INTERACTIVE DEMO (optional):
   powershell -ExecutionPolicy Bypass -File .\run_app.ps1
   then open http://localhost:8501, upload two face images, move the alpha slider.
   (Requires: pip install -r requirements.txt. models/face_landmarker.task included.)

Dataset: face crops from Labeled Faces in the Wild (LFW).
