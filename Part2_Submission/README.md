# DMorphNet — Face Morphing Detection (Part 2 Submission)

Implementation of the article:

> **Gawade, S.S., Gujar, A.V., Harekar, A.R., Jadhav, A.S., Teli, S.R.:
> "DMorphNet: Face Morphing Detection Using Generative Adversarial Networks
> and EfficientNet-B6."** In: B. Singh et al. (eds.), *Proceedings of the
> International Conference on Advances in Computing Technology and Artificial
> Intelligence (COMPUTATIA 2026)*, Atlantis Highlights in Intelligent
> Systems 18, pp. 277–291 (2026).
> DOI: [10.2991/978-94-6239-713-2_20](https://doi.org/10.2991/978-94-6239-713-2_20)
> — Open Access (CC BY-NC 4.0). Full article included here as
> `DMorphNet_Gawade2026_COMPUTATIA_article.pdf`.

The task: detect whether a face image is **real** or a **morph** (two
identities blended into one) using **EfficientNet-B6** deep features and an
**SVM** classifier, exactly as proposed in the paper.

## Pipeline (as in the paper, Sec. 3)

```text
face → resize 528 + CLAHE → EfficientNet-B6 (pretrained, GAP) → 2304-d vector → SVM (RBF) → Real / Morph
```

## Run

**On your own PC (Windows):**

```powershell
python -m pip install tensorflow kagglehub mediapipe opencv-python scikit-learn tqdm matplotlib pandas notebook
python -m notebook DMorphNet.ipynb
```

Then *Run All* cells top to bottom. The notebook automatically moves its
working directory to an ASCII path (`%LOCALAPPDATA%\dmorphnet_workdir`), so
it works even if this folder's path contains non-English characters.

**In Google Colab:** upload `DMorphNet.ipynb`, enable GPU
(Runtime → Change runtime type → GPU), and Run All.

`DEMO = True` (default) runs a fast subset (900 real / 700 morph) that still
produces real results in minutes; set `DEMO = False` for the full-scale run
(24,000 real / 21,000 morph — the paper's dataset scale, needs a GPU).

## Adaptations from the paper

- **Real faces:** FFHQ (256px Kaggle mirror) instead of the paper's
  FaceMorph_EClub_Task set (not publicly redistributable).
- **Morphs:** landmark morphing (Delaunay + affine warp + seamless clone)
  instead of the paper's "AI FaceSwap" tool, so the whole dataset can be
  regenerated from code.
- Everything else follows the paper: 528×528 + CLAHE preprocessing,
  frozen pretrained EfficientNet-B6 as a pure feature extractor
  (classification head removed, Global Average Pooling → 2304-d),
  SVM (RBF) classifier, and threshold optimization on a validation set.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Paper (Gawade et al., Table 2) | 89.9% | 0.90 | 0.90 | 0.90 | 0.965 |
| This implementation | **91.7%** | 0.951 | 0.867 | 0.907 | **0.967** |

The notebook also reproduces the paper's comparative analysis (EfficientNet
variants / classifier heads) with 6 step-by-step improvement experiments,
threshold optimization (paper Sec. 4.4), and a real-time single-image
prediction demo (paper Sec. 4.5).

## Files

- `DMorphNet.ipynb` — the notebook (single, clean, end-to-end)
- `DMorphNet_Gawade2026_COMPUTATIA_article.pdf` — the implemented article
