"""Render a two-page morphing submission report, with key figures and code snippets embedded."""
import os
from fpdf import FPDF

BASE = os.getcwd()
OUT = os.path.join(BASE, "outputs")

TITLE_SIZE, H_SIZE, BODY_SIZE, SMALL_SIZE = 14, 10.5, 8.8, 8.0
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
    pdf.multi_cell(0, 6.5, text)
    pdf.ln(1.5)


def h2(pdf, text):
    pdf.set_font("Helvetica", "B", H_SIZE)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(0.8)


def body(pdf, text):
    pdf.set_font("Helvetica", size=BODY_SIZE)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 4.2, text)
    pdf.ln(1.2)


def bullets(pdf, items):
    pdf.set_font("Helvetica", size=BODY_SIZE)
    for it in items:
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, 4.2, f"-  {it}")
    pdf.ln(1.2)


def code_block(pdf, lines, title=None):
    if title:
        pdf.set_font("Helvetica", "BI", 7.8)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(180, 4.0, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", size=7.5)
    pdf.set_fill_color(244, 246, 249)
    pdf.set_text_color(25, 25, 25)
    for line in lines:
        pdf.set_x(MARGIN + 2)
        pdf.cell(180 - 4, 3.7, f"  {line}", border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.8)


def caption(pdf, text):
    pdf.set_font("Helvetica", "I", SMALL_SIZE)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 3.8, text, align="C")
    pdf.ln(1.5)


pdf = Report(format="A4")
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(MARGIN, 14, MARGIN)
pdf.set_title("Face Morphing - Final Project Report")

# ---------------- Page 1: Sections 1 to 4 with Code Snippets + First Showcase Pair ----------------
pdf.add_page()
h1(pdf, "Face Morphing - Final Project Report")

h2(pdf, "1. Method Used")
body(pdf, "We implement a classical landmark-based face morphing pipeline (Beier-Neely / Delaunay approach, widely studied in face-morphing-attack literature such as Ferrara et al., IJCB 2014). Given two bona fide face photos of different identities (Subject A and Subject B) from Labeled Faces in the Wild (LFW), the pipeline detects dense facial landmarks, aligns corresponding points, computes a Delaunay triangulation on the averaged target shape, performs piecewise-affine warping on every triangle patch, and cross-dissolves the pixel intensities.")
code_block(pdf, [
    "def morph_images(img1, pts1, img2, pts2, alpha, triangles=None):",
    "    pts_morph = (1 - alpha) * pts1 + alpha * pts2  # Averaged target shape",
    "    triangles = delaunay_triangulation(pts_morph, (w, h))",
    "    # Warp each triangle from img1 & img2 into the target shape, then blend:",
    "    morphed = (1 - alpha) * warped1 + alpha * warped2",
], title="src/morph.py - Core pipeline overview:")

h2(pdf, "2. Landmark Alignment Strategy")
body(pdf, "We obtain 478 3D facial landmarks using MediaPipe FaceLandmarker (with 3x bicubic upscaling of the 125x94 LFW crops for robust tracking). Because MediaPipe tracks only the face mesh, we append 8 fixed boundary points (the 4 corners + 4 edge midpoints of the frame) to both landmark arrays. This ensures that background and frame borders participate in the continuous Delaunay triangulation and warping instead of shearing. Correspondence is purely index-based (point i on Face A corresponds to point i on Face B), and one single Delaunay topology computed on the convex combination shape is valid simultaneously for both faces.")
code_block(pdf, [
    "def get_landmarks(img_bgr, upscale=3):",
    "    # ... MediaPipe detection of 478 face landmarks ...",
    "    pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]])",
    "    # Append 8 boundary points so the full frame participates in the warp:",
    "    pts = np.vstack([pts, _boundary_points(w, h)])",
    "    return pts, img_bgr",
], title="src/landmarks.py - Boundary extension & correspondence:")

h2(pdf, "3. Interpolation Technique")
body(pdf, "We combine geometric and photometric interpolation. For each Delaunay triangle, we compute the affine transform matrix mapping source vertices to target vertices (cv2.getAffineTransform) and warp the source patch (cv2.warpAffine). A filled polygon mask composites the warped patch cleanly into the target canvas. The two warped images (warped1 and warped2) are then linearly cross-dissolved with weights (1-alpha, alpha).")
code_block(pdf, [
    "def _warp_triangle(src_img, dst_img, tri_src, tri_dst):",
    "    mat = cv2.getAffineTransform(np.float32(tri_src_rect), np.float32(tri_dst_rect))",
    "    warped = cv2.warpAffine(src_patch, mat, (r_dst[2], r_dst[3]), flags=cv2.INTER_LINEAR)",
    "    cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), (1.0, 1.0, 1.0), cv2.LINE_AA)",
    "    dst_slice[:] = dst_slice * (1 - mask_c) + warped_c * mask_c",
], title="src/morph.py - Piecewise-affine triangle warp & compositing:")

h2(pdf, "4. Observations and Limitations")
bullets(pdf, [
    "At alpha=0.5, morphs successfully blend both identities into a convincing, single face (demonstrating vulnerability to morphing attacks).",
    "Mismatched pose/expression across pairs (e.g., open mouth with teeth vs. closed lips) creates visible ghosting or double lips.",
    "Accessories unique to one subject (such as eyeglasses or hair bangs) fade semi-transparently rather than retaining solid physical structure.",
    "We use linear cross-dissolve rather than Poisson/gradient-domain cloning, meaning lighting and skin-tone differences are not color-corrected across the subjects.",
])

# First showcase pair at the bottom of Page 1 (w=112 mm, centered)
h2(pdf, "5. Showcase: 5 Pairs of Original Faces and Morphs (alpha=0.5)")
pdf.image(os.path.join(OUT, "showcase", "pair000_triptych.png"), x=(210-112)/2, w=112)

# ---------------- Page 2: Remaining 4 Showcase Pairs + Figure Caption ----------------
pdf.add_page()

pdf.image(os.path.join(OUT, "showcase", "pair003_triptych.png"), x=(210-126)/2, w=126)
pdf.ln(1.8)
pdf.image(os.path.join(OUT, "showcase", "pair004_triptych.png"), x=(210-126)/2, w=126)
pdf.ln(1.8)
pdf.image(os.path.join(OUT, "showcase", "pair007_triptych.png"), x=(210-126)/2, w=126)
pdf.ln(1.8)
pdf.image(os.path.join(OUT, "showcase", "pair009_triptych.png"), x=(210-126)/2, w=126)
pdf.ln(2.5)

caption(pdf, "Figure 1: Five complete triptychs of original bona fide face pairs from LFW ('pair000' on Page 1; 'pair003', 'pair004', 'pair007', and 'pair009' above) and their resulting attack morphs at alpha=0.5. Each morph seamlessly fuses the facial geometry and appearance of Subject A and Subject B using our Delaunay triangulation and piecewise-affine pipeline.")

out_path = os.path.join(BASE, "submission_report.pdf")
pdf.output(out_path)
print(f"wrote {out_path} (Total pages: {pdf.page_no()})")




