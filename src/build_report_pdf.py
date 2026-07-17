"""Render a two-page morphing submission report, with the key figures embedded."""
import os
from fpdf import FPDF

BASE = os.getcwd()
OUT = os.path.join(BASE, "outputs")

TITLE_SIZE, H_SIZE, BODY_SIZE, SMALL_SIZE = 15, 11.5, 9.2, 8.2
MARGIN = 15


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def h1(pdf, text):
    pdf.set_font("Helvetica", "B", TITLE_SIZE)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 7, text)
    pdf.ln(1)


def h2(pdf, text):
    pdf.set_font("Helvetica", "B", H_SIZE)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 6, text)
    pdf.ln(0.5)


def body(pdf, text):
    pdf.set_font("Helvetica", size=BODY_SIZE)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 4.6, text)
    pdf.ln(1)


def bullets(pdf, items):
    pdf.set_font("Helvetica", size=BODY_SIZE)
    for it in items:
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, 4.6, f"-  {it}")
    pdf.ln(1)


def caption(pdf, text):
    pdf.set_font("Helvetica", "I", SMALL_SIZE)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4, text, align="C")
    pdf.ln(2)


pdf = Report(format="A4")
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(MARGIN, 14, MARGIN)
pdf.set_title("Face Morphing - Final Project Report")

# ---------------- Page 1: Part 1, sections 1-4 ----------------
pdf.add_page()
h1(pdf, "Face Morphing - Final Project Report")

h2(pdf, "1. Method")
body(pdf, "We implement classical landmark-based face morphing: given two face "
          "images of different identities, we (a) detect dense facial landmarks "
          "on each face, (b) establish point correspondence via matching "
          "landmark indices, (c) triangulate the averaged landmark shape with "
          "Delaunay triangulation, (d) warp each source image's triangles into "
          "the corresponding triangles of the target (morph) shape with a "
          "piecewise-affine transform, and (e) cross-dissolve (linearly blend) "
          "the two warped images. This is the same family of technique used to "
          "generate face-morphing-attack images in the biometrics literature "
          "(e.g. Ferrara, Franco & Maltoni, \"The Magic Passport,\" IJCB 2014).")

h2(pdf, "2. Inputs")
body(pdf, "The interactive dashboard accepts two user-provided face images. "
          "For reliable landmark detection and a clean morph, use clear, "
          "front-facing portraits with one visible face per image.")

h2(pdf, "3. Landmark detection & alignment strategy")
body(pdf, "Landmarks are obtained with MediaPipe's FaceLandmarker (a 478-point "
          "3-D face mesh). Because MediaPipe only tracks the "
          "face region, 8 fixed boundary points (image corners + edge "
          "midpoints) are appended to both landmark sets so the background "
          "also participates in the warp instead of only the face.\n"
          "Correspondence between the two faces is purely index-based: "
          "landmark i on face A corresponds to landmark i on face B, since "
          "both come from the same 478-point model with a fixed topology -- "
          "no separate registration/ICP step is needed. The morph target "
          "shape at blend factor alpha is the per-point convex combination "
          "(1-alpha)*ptsA + alpha*ptsB, and the Delaunay triangulation used "
          "for warping is computed on this averaged shape (not on either "
          "source shape individually), so one triangle topology is "
          "simultaneously valid for warping both source images.")

h2(pdf, "4. Interpolation technique")
body(pdf, "For a chosen alpha in [0,1], each of the ~950 Delaunay triangles is "
          "affine-warped from face A, and separately from face B, into its "
          "position in the alpha-blended target shape (cv2.getAffineTransform "
          "+ warpAffine); the two warped triangle patches are then "
          "cross-dissolved with weights (1-alpha, alpha) and composited using "
          "a filled-polygon mask. This reproduces both the geometric "
          "interpolation (the shape warps smoothly with alpha) and the "
          "photometric interpolation (pixel intensities blend smoothly with "
          "alpha) that a real morphing attack relies on. We render the "
          "standard attack morph at alpha=0.5 for every pair, plus an "
          "alpha = 0, 0.25, 0.5, 0.75, 1 strip for a subset, to visualize the "
          "interpolation continuum (Figure 1).")

pdf.image(os.path.join(OUT, "strips", "pair007_strip.png"), w=180)
caption(pdf, "Figure 1: interpolation strip at alpha = 0, 0.25, 0.5, 0.75, 1.")

# ---------------- Page 2: Part 1 section 5 + figure ----------------
pdf.add_page()
h2(pdf, "5. Observations & limitations")
bullets(pdf, [
    "At alpha=0.5 the morphs are visually convincing single identities that "
    "plausibly resemble both source subjects -- exactly the property that "
    "makes morphing attacks a biometric security threat.",
    "Quality depends heavily on how well the two source photos agree in "
    "pose and expression: pairs with mismatched mouth state (one subject "
    "smiling with teeth, the other neutral) show visible ghosting around "
    "the mouth.",
    "A feature present in only one subject (e.g. eyeglasses) fades in/out "
    "across alpha rather than looking physically consistent -- a classic, "
    "visible morphing artifact.",
    "Small seams appear near the image border, since background pixels far "
    "from any landmark are only weakly constrained by the triangulation.",
    "We use simple cross-dissolve rather than Poisson/gradient-domain "
    "blending, so skin-tone/lighting differences between the two subjects "
    "are not colour-corrected -- a known limitation relative to "
    "production-grade morphing tools.",
    "Source image resolution (125x94 before upscaling) caps the fine "
    "detail achievable in the final morph.",
])

pdf.image(os.path.join(OUT, "showcase", "pair007_triptych.png"), w=180)
caption(pdf, "Figure 2: subject A, morph (alpha=0.5), subject B -- one of the "
             "five showcased pairs (see outputs/showcase/ for all five, and "
             "outputs/pairs/ + outputs/morphs/ for every generated pair).")

out_path = os.path.join(BASE, "submission_report.pdf")
pdf.output(out_path)
print("wrote", out_path)
