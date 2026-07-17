"""Render a detailed step-by-step ML SDLC (Software Development Life Cycle) technical report with code snippets."""
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
        self.cell(0, 8, f"ML SDLC Technical Report - Page {self.page_no()}", align="C")


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
    pdf.set_font("Courier", size=7.3)
    pdf.set_fill_color(244, 246, 249)
    pdf.set_text_color(25, 25, 25)
    for line in lines:
        pdf.set_x(MARGIN + 2)
        pdf.cell(180 - 4, 3.6, f"  {line}", border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.8)


pdf = Report(format="A4")
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(MARGIN, 14, MARGIN)
pdf.set_title("Machine Learning SDLC - Step-by-Step Technical Guide")

# ================= Page 1: ML SDLC Overview + Phase 1 & Phase 2 =================
pdf.add_page()
h1(pdf, "Machine Learning SDLC - Step-by-Step Architecture & Code Mapping")
body(pdf, "This report breaks down the Machine Learning Software Development Life Cycle (ML SDLC / AI Engineering Lifecycle) implemented in the Face Morph Studio repository. Every phase of our computer vision pipeline corresponds precisely to industry-standard ML engineering practices: data ingestion, feature/landmark extraction, algorithmic modeling, batch verification, and interactive production deployment.")

h2(pdf, "Phase 1: Data Acquisition & Ingestion (Data Engineering)")
body(pdf, "In the initial SDLC stage, raw domain data must be sourced, filtered, and standardized. In our project, we fetch pre-aligned face images from Labeled Faces in the Wild (LFW) using scikit-learn. We specifically filter the dataset for 'different person' pairs (target == 0), convert the RGB float arrays [0, 1] to standard BGR uint8 [0, 255] images, and persist deterministic pairs to disk using a seeded random generator (seed=42) to ensure strict experimental reproducibility.")
code_block(pdf, [
    "def main():",
    "    lfw = fetch_lfw_pairs(subset='train', color=True, resize=1.0, funneled=True)",
    "    pairs, targets = lfw.pairs, lfw.target",
    "    diff_idx = np.where(targets == 0)[0]  # Filter for different-identity pairs only",
    "    rng = np.random.default_rng(42)       # Seeded RNG for reproducibility",
    "    chosen = rng.choice(diff_idx, size=min(60, len(diff_idx)), replace=False)",
    "    for rank, idx in enumerate(chosen):",
    "        imgA = (pairs[idx, 0] * 255).astype(np.uint8)",
    "        imgB = (pairs[idx, 1] * 255).astype(np.uint8)",
    "        imwrite(os.path.join(out_dir, f'pair{rank:03d}_A.png'), cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR))",
], title="src/fetch_data.py - Data sourcing, filtering, and normalization:")

h2(pdf, "Phase 2: Data Preprocessing & Feature Extraction (Keypoint Engineering)")
body(pdf, "Before geometric modeling can occur, raw pixel arrays must be preprocessed and mapped to structured feature representations. In face morphing, feature extraction corresponds to tracking dense 3D facial mesh keypoints. To make MediaPipe robust on small LFW crops (125x94 px), we apply 3x bicubic upscaling. Crucially, because MediaPipe only tracks the face mask, we engineer 8 synthetic boundary points (image corners + edge midpoints) and append them to the landmark array so the background participates continuously in the warp.")
code_block(pdf, [
    "def get_landmarks(img_bgr, upscale=3):",
    "    # 1. Upscale for reliable MediaPipe inference on small face crops:",
    "    img_bgr = cv2.resize(img_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)",
    "    result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(...)))",
    "    if not result.face_landmarks: return None, None",
    "    # 2. Extract 478 3D facial mesh points (scaled to upscaled pixel dims):",
    "    pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float64)",
    "    # 3. Feature engineering: append 8 frame boundary anchor points:",
    "    pts = np.vstack([pts, _boundary_points(w, h)])",
    "    return pts, img_bgr",
], title="src/landmarks.py - Upscaling, mesh detection, and boundary feature appending:")

# ================= Page 2: Phase 3 & Phase 4 =================
pdf.add_page()
h2(pdf, "Phase 3: Model Architecture & Algorithmic Core (Geometric & Photometric Engine)")
body(pdf, "The core transformation algorithm combines topological graph generation and piecewise-affine warping. First, we compute the target shape at blend factor alpha via convex combination: pts_morph = (1-alpha)*ptsA + alpha*ptsB. Next, we construct a Delaunay triangulation graph (cv2.Subdiv2D) on this averaged shape. For each triangle patch, we derive the affine transform matrix (cv2.getAffineTransform), warp the pixel patch (cv2.warpAffine), and composite it using a filled polygon mask. Finally, warped frames are linearly cross-dissolved.")
code_block(pdf, [
    "def morph_images(img1, pts1, img2, pts2, alpha, triangles=None):",
    "    # 1. Averaged target shape where topological triangulation is established:",
    "    pts_morph = (1 - alpha) * pts1 + alpha * pts2",
    "    if triangles is None:",
    "        triangles = delaunay_triangulation(pts_morph, (w, h))",
    "    # 2. Piecewise-affine warping of each triangle index triplet (i, j, k):",
    "    for (i, j, k) in triangles:",
    "        _warp_triangle(img1f, warped1, [pts1[i], pts1[j], pts1[k]], [pts_morph[i], ...])",
    "        _warp_triangle(img2f, warped2, [pts2[i], pts2[j], pts2[k]], [pts_morph[i], ...])",
    "    # 3. Photometric cross-dissolving across both warped source images:",
    "    morphed = (1 - alpha) * warped1 + alpha * warped2",
    "    return np.clip(morphed, 0, 255).astype(np.uint8), triangles",
], title="src/morph.py - Delaunay graph topology & piecewise-affine cross-dissolving:")

h2(pdf, "Phase 4: Batch Verification, QA Auditing & Artifact Pipeline (Testing)")
body(pdf, "In production ML engineering, individual algorithms must be validated in batch mode across extensive datasets. Our batch pipeline iterates over all candidate pairs, safely catching detection anomalies (skipping pairs where face mesh tracking fails without crashing), writing standardized attack morphs (alpha=0.5), building 5-step interpolation strips (alpha = 0, 0.25, 0.5, 0.75, 1.0), and logging QA metrics to a JSON audit manifest.")
code_block(pdf, [
    "# Batch pipeline across N pairs with automated anomaly skipping & audit logging:",
    "for k, pid in enumerate(pair_ids):",
    "    ptsA, upA = get_landmarks(imgA); ptsB, upB = get_landmarks(imgB)",
    "    if ptsA is None or ptsB is None: n_fail += 1; continue  # QA exception handling",
    "    morphed, triangles = morph_images(upA, ptsA, upB, ptsB, alpha=0.5)",
    "    imwrite(os.path.join(OUT_MORPHS, f'{pid}_morph.png'), morphed)",
    "    manifest.append(pid); n_ok += 1",
    "with open(os.path.join(BASE, 'outputs', 'morph_manifest.json'), 'w') as f:",
    "    json.dump(manifest, f, indent=2)",
], title="src/generate_morphs.py - Batch verification, exception handling, and manifest logging:")

# ================= Page 3: Phase 5 & System Summary =================
pdf.add_page()
h2(pdf, "Phase 5: Production Deployment & Serving (Interactive Front-End)")
body(pdf, "The final stage of the ML SDLC exposes the core algorithmic engine through a live interactive interface. Built with Streamlit, our serving tier handles real-time user uploads and slider events. To ensure low latency during interactive alpha scrubbing, heavy feature extraction is wrapped in @st.cache_data decorators. Furthermore, the serving layer introduces advanced post-processing: isolating the facial region via convex hull and compositing it seamlessly into Subject A's or Subject B's original hair/background using Poisson blending (cv2.seamlessClone).")
code_block(pdf, [
    "@st.cache_data(show_spinner=False)",
    "def prepare_pair(file_bytes_a, file_bytes_b, target_margin=TARGET_MARGIN, size=SQUARE_SIZE):",
    "    # Cached face scale alignment and MediaPipe keypoint detection for low-latency serving",
    "    return pts_a, up_a, pts_b, up_b",
    "",
    "def render_morph(up_a, pts_a, up_b, pts_b, alpha, frame_from, triangles=None):",
    "    # Advanced production post-processing: Poisson seamless cloning of blended face into photo frame",
    "    hull = cv2.convexHull(pts_m[:478].astype(np.int32))",
    "    cv2.fillConvexPoly(mask, hull, 255)",
    "    try:",
    "        out = cv2.seamlessClone(blend8, frame8, mask, (x+bw//2, y+bh//2), cv2.NORMAL_CLONE)",
    "    except cv2.error:",
    "        # Fallback for tightly-cropped edge boundary cases: feathered Gaussian blending",
    "        soft = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), np.uint8)), (31, 31), 0)",
    "        out = np.clip(blend * (soft/255.0) + frame * (1 - soft/255.0), 0, 255).astype(np.uint8)",
    "    return out, triangles",
], title="src/app.py - Caching optimization & Poisson seamless cloning for production serving:")

h2(pdf, "Summary of ML SDLC Mapping Across Repository Modules")
bullets(pdf, [
    "src/fetch_data.py -> Phase 1: Data Sourcing, Filtering (different person split), and BGR Normalization.",
    "src/landmarks.py -> Phase 2: Feature Engineering (3x upscaling, 478-pt MediaPipe mesh, 8 boundary anchors).",
    "src/morph.py -> Phase 3: Core Algorithmic Engine (Delaunay triangulation, piecewise-affine warp, cross-dissolve).",
    "src/generate_morphs.py & build_showcase.py -> Phase 4: Batch Verification, Exception Handling, QA Audit Logging.",
    "src/app.py -> Phase 5: Production Deployment (Streamlit UI, caching layer, Poisson seamless cloning post-processing).",
])

out_path = os.path.join(BASE, "ml_sdlc_report.pdf")
pdf.output(out_path)
print(f"wrote {out_path} (Total pages: {pdf.page_no()})")
