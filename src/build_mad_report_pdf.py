"""Render the Part 2 morphing-attack-detection report: method, dataset,
evaluation protocol, results and limitations, with key figures and code
snippets embedded."""
import json
import os
from fpdf import FPDF

BASE = os.getcwd()
OUT = os.path.join(BASE, "outputs", "mad")

TITLE_SIZE, H_SIZE, BODY_SIZE, SMALL_SIZE = 14, 10.5, 8.8, 8.0
MARGIN = 15

with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as f:
    M = json.load(f)
with open(os.path.join(OUT, "nn_metrics.json"), encoding="utf-8") as f:
    NN = json.load(f)


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


def table(pdf, rows, col_widths):
    pdf.set_font("Helvetica", "B", BODY_SIZE)
    pdf.set_fill_color(230, 233, 238)
    for w, cell in zip(col_widths, rows[0]):
        pdf.cell(w, 5.5, cell, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", size=BODY_SIZE)
    for row in rows[1:]:
        for w, cell in zip(col_widths, row):
            pdf.cell(w, 5.5, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(2)


pdf = Report(format="A4")
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(MARGIN, 14, MARGIN)
pdf.set_title("Face Morphing Attack Detection - Part 2 Report")

# ---------------- Page 1 ----------------
pdf.add_page()
h1(pdf, "Part 2: Face Morphing Attack Detection")

h2(pdf, "1. Chosen Article")
body(pdf, "Venkatesh, S.; Raghavendra, R.; Raja, K.; Busch, C. \"Single Image Face Morphing Attack Detection "
          "Using Ensemble of Features.\" Proc. IEEE 23rd International Conference on Information Fusion (FUSION), "
          "2020. This is a Single-image Morphing Attack Detection (S-MAD) method: it decides bona fide vs. morphed "
          "from one image alone, with no trusted live-capture reference -- the right fit for our setup, since Part 1 "
          "only produced the two source photos that were morphed together, not a separate trusted capture of either "
          "subject.")

h2(pdf, "2. Method (as implemented)")
body(pdf, "Each face image is represented in two color spaces (YCbCr, HSV, 6 channel images total). Each channel "
          "gets a 3-level Laplacian pyramid to expose morphing residue at multiple scales (18 scale-space images per "
          "face). Three texture descriptors are computed on every scale-space image and concatenated per descriptor "
          "type across all 18: Local Binary Patterns (LBP, block-wise), Histogram of Oriented Gradients (HOG), and "
          "Binarized Statistical Image Features (BSIF). Each of the 3 resulting streams is classified independently "
          "by a Collaborative Representation Classifier (CRC): a probe vector is reconstructed as a ridge-regularized "
          "combination of each class's training vectors, and the class with lower reconstruction residual wins. The "
          "per-stream scores are z-score normalized against their own training distribution, then fused by summation.")
code_block(pdf, [
    "def scale_space_images(img_bgr):  # 6 color channels x 3 Laplacian levels",
    "    return [_resize(level) for ch in to_color_channels(img_bgr)",
    "                           for level in laplacian_pyramid(ch)]",
    "",
    "class CRCClassifier:",
    "    def fit(self, X, y):        # dictionary D_c = training vectors of class c",
    "        D = Xn[y == c].T; self.proj_[c] = inv(D.T @ D + lam*I) @ D.T",
    "    def score(self, X):         # residual(bona fide) - residual(morph)",
    "        return [self._residual(x, 0) - self._residual(x, 1) for x in Xn]",
], title="src/mad_features.py, src/mad_crc.py -- pipeline core:")

h2(pdf, "3. Adaptations from the Paper")
bullets(pdf, [
    "BSIF normally uses a filter bank pre-learned via ICA on ~50,000 natural-image patches (Kannala & Rahtu, 2012). "
    "We do not have that corpus, so we learn a smaller ICA filter bank (6 filters, 5x5) directly from each "
    "cross-validation fold's own bona fide training images instead -- same principle, much smaller and fold-local "
    "training data.",
    "The paper evaluates digital and print-scanned morphs on 1,309 bona fide / 2,608 morphed images with a single "
    "fixed identity-disjoint train/test split. Our Part 1 dataset has 116 bona fide / 58 morphed digital images only, "
    "so we use 5-fold pair-disjoint cross-validation (GroupKFold on the source-pair id) to get an out-of-fold score "
    "for every image while still preventing any image pair from straddling train and test.",
    "We z-score each descriptor stream's scores against its own training distribution before the sum-rule fusion; "
    "the original paper does not state whether it normalizes before summing, and skipping this step let BSIF's very "
    "high-dimensional (73,728-d) residual scale dominate the fused score in our own initial run.",
])

h2(pdf, "4. Dataset and Evaluation Protocol")
body(pdf, f"Bona fide class: the {M['n_bona_fide']} original LFW source images (Subject A and B) from Part 1's "
          f"{M['n_pair_groups']} pairs. Attack class: the {M['n_morph']} corresponding alpha=0.5 morphs. Evaluated "
          f"with {M['n_folds']}-fold pair-disjoint cross-validation and reported with the ISO/IEC 30107-3 metrics: "
          "APCER (attacks classified as bona fide), BPCER (bona fide classified as attacks), and D-EER (operating "
          "point where the two are equal).")

# ---------------- Page 2 ----------------
pdf.add_page()

h2(pdf, "5. Results")
table(pdf, [
    ["Configuration", "D-EER (%)"],
    ["LBP stream alone", f"{M['per_stream_d_eer_percent']['lbp']:.2f}"],
    ["HOG stream alone", f"{M['per_stream_d_eer_percent']['hog']:.2f}"],
    ["BSIF stream alone", f"{M['per_stream_d_eer_percent']['bsif']:.2f}"],
    ["Fused ensemble (LBP+HOG+BSIF)", f"{M['d_eer_percent']:.2f}"],
], col_widths=[110, 60])
body(pdf, f"At the ISO/IEC 30107-3 operating points: BPCER = {M['bpcer_at_apcer5_percent']:.2f}% at APCER = 5%, and "
          f"BPCER = {M['bpcer_at_apcer10_percent']:.2f}% at APCER = 10%.")

pdf.image(os.path.join(OUT, "det_curve.png"), x=(210 - 110) / 2, w=110)
caption(pdf, "Figure 1: DET curve of the fused ensemble score, pair-disjoint 5-fold cross-validation "
             "(both axes log-scaled per ISO/IEC 30107-3 convention).")

h2(pdf, "6. Observations and Limitations")
bullets(pdf, [
    "The fused ensemble (D-EER 17.2%) is noticeably weaker than the paper's own reported 5.99-5.64% D-EER, mainly "
    "because our training set is two orders of magnitude smaller (174 vs 3,900+ images) and identity-disjoint "
    "cross-validation on 58 pairs gives coarse-grained, high-variance error estimates.",
    "BSIF performed close to chance alone (43.1% D-EER) despite contributing to the best fused score, which we "
    "attribute to the filter-bank substitution (Adaptation #1): fold-local ICA on a few dozen bona fide images is a "
    "much weaker texture prior than filters learned on 50,000 diverse natural-image patches.",
    "LBP and HOG carried almost all of the usable signal (~19% D-EER each); fusing all three still beat either "
    "stream alone, showing the ensemble/sum-fusion idea holds even when one stream is individually weak.",
    "Our attack set only contains alpha=0.5 morphs from a single morphing pipeline (our own Part 1 Delaunay/warp "
    "code); a detector trained here is not shown to generalize to other morphing tools or alpha values, mirroring "
    "the generalization gap this literature repeatedly flags as an open problem.",
])

# ---------------- Page 3 ----------------
pdf.add_page()

h2(pdf, "7. Neural Network Fusion Layer")
body(pdf, "The paper combines the three descriptor-stream scores with a fixed, hand-picked sum rule. We additionally "
          "replace that fixed rule with the simplest possible neural network -- a single trainable neuron (logistic "
          "regression) -- that learns how much to weight each stream instead: "
          "z = w_lbp*lbp + w_hog*hog + w_bsif*bsif + b, p = sigmoid(z), trained by full-batch gradient descent on the "
          "binary cross-entropy loss. Its 3 inputs are the same z-scored, out-of-fold CRC scores from Section 5/6 -- "
          "already leakage-free -- so this experiment only needed one further pair-disjoint 80/20 train/test split "
          f"({NN['n_train']} training images / {NN['n_test']} test images) to fit and evaluate the neuron itself.")
code_block(pdf, [
    "class LogisticNeuron:",
    "    def fit(self, X, y):          # X = [lbp_score, hog_score, bsif_score] per image",
    "        for _ in range(epochs):",
    "            p = sigmoid(Xn @ self.weights + self.bias)",
    "            loss = -mean(y*log(p) + (1-y)*log(1-p))      # binary cross-entropy",
    "            self.weights -= lr * Xn.T @ (p - y) / n       # gradient descent",
    "            self.bias    -= lr * mean(p - y)",
], title="src/mad_nn.py -- the fusion neuron:")

table(pdf, [
    ["Metric", "Value"],
    ["Training accuracy", f"{NN['train_accuracy_percent']:.2f}%"],
    ["Test accuracy", f"{NN['test_accuracy_percent']:.2f}%"],
    ["Final training cross-entropy loss", f"{NN['final_train_cross_entropy_loss']:.4f}"],
    ["Weight - LBP stream", f"{NN['weights']['lbp']:.3f}"],
    ["Weight - HOG stream", f"{NN['weights']['hog']:.3f}"],
    ["Weight - BSIF stream", f"{NN['weights']['bsif']:.3f}"],
    ["Bias", f"{NN['bias']:.3f}"],
], col_widths=[110, 60])

pdf.image(os.path.join(OUT, "nn_loss_curve.png"), x=(210 - 100) / 2, w=100)
caption(pdf, "Figure 2: Binary cross-entropy training loss of the fusion neuron over "
             f"{NN['epochs']} gradient-descent epochs (learning rate {NN['learning_rate']}).")

body(pdf, "The learned weights corroborate Section 6's finding purely from the training data, with no hand-tuning: "
          "LBP and HOG get large positive weights (the neuron relies on them heavily), while BSIF's weight is small "
          "and negative -- the network learned on its own that the BSIF stream carries little useful signal, "
          "consistent with its near-chance 43.1% standalone D-EER.")

h2(pdf, "8. Conclusion")
body(pdf, "We reimplemented the ensemble-of-features S-MAD pipeline from Venkatesh et al. (FUSION 2020) end-to-end "
          "on our own bona fide/morph dataset from Part 1: color-space expansion, Laplacian-pyramid scale space, "
          "LBP/HOG/BSIF descriptors, per-stream P-CRC classification, and sum-rule fusion, evaluated with the "
          "ISO/IEC 30107-3 metrics the field standardizes on. The pipeline detects our own morphs meaningfully "
          "better than chance (D-EER 17.2%), with the expected accuracy gap to the original paper explained by "
          "dataset scale and our BSIF filter-bank substitution, both documented above. We further trained a small "
          f"neural network (a single logistic-regression neuron) to learn the stream-fusion weights directly from "
          f"data instead of a fixed sum rule, reaching {NN['test_accuracy_percent']:.1f}% test accuracy and "
          "independently confirming, through its learned weights, which descriptor streams actually carry signal.")

out_path = os.path.join(BASE, "submission_report_part2.pdf")
pdf.output(out_path)
print(f"wrote submission_report_part2.pdf (Total pages: {pdf.page_no()})")
