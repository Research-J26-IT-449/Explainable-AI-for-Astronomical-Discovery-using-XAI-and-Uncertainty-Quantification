"""
generate_corrected_presentation.py
==================================
Generates the fully audited and corrected PowerPoint presentation:
  presentation/STELLAR_SPECTRA_PP1_PRESENTATION_CORRECTED.pptx
and corrected speaker notes:
  presentation/PRESENTATION_SPEAKER_NOTES_CORRECTED.docx

Incorporates all 15 slide corrections from the independent audit:
- Exact CNN-BiLSTM architecture from model_architectures.py (no attention)
- Corrected metrics and honest classical baseline comparison
- Gradient SHAP (shap.GradientExplainer) terminology
- 6 verified atomic line matches (O I, H Br14, N I, Si I, Mg I, O I)
- Honest p=0.538 Monte Carlo conclusion (exploratory / not statistically significant)
- Descriptive AOPC faithfulness (6.30x ratio) & synthetic noise stability
- Documented NPY / FITS contracts and live per-upload SHAP
- 60/60 passing automated assertions
"""

import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import docx

ROOT = Path(__file__).resolve().parent

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

DARK_BLUE = RGBColor(15, 32, 67)
LIGHT_BLUE = RGBColor(41, 128, 185)
WHITE = RGBColor(255, 255, 255)
DARK_GRAY = RGBColor(40, 40, 40)
LIGHT_GRAY = RGBColor(245, 247, 250)
ACCENT_ORANGE = RGBColor(230, 126, 34)

speaker_notes = []

def add_header(slide, title_text, category_text="IT4010 RESEARCH PROJECT | PP1 ASSESSMENT"):
    top_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.1))
    top_shape.fill.solid()
    top_shape.fill.fore_color.rgb = DARK_BLUE
    top_shape.line.color.rgb = DARK_BLUE

    tf = top_shape.text_frame
    tf.margin_left = Inches(0.8)
    tf.margin_top = Inches(0.15)
    
    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_ORANGE
    
    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

def add_content_box(slide, left, top, width, height, title, bullet_points):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY
    shape.line.color.rgb = LIGHT_BLUE
    shape.line.width = Pt(1.5)
    
    tf = shape.text_frame
    tf.margin_left = Inches(0.3)
    tf.margin_right = Inches(0.3)
    tf.margin_top = Inches(0.2)
    
    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(16)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_BLUE
    
    for bp in bullet_points:
        p = tf.add_paragraph()
        p.text = f"• {bp}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(6)

# ── Slide 1: Title Slide ──────────────────────────────────────────────────────
s1 = prs.slides.add_slide(blank_layout)
bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
bg1.fill.solid(); bg1.fill.fore_color.rgb = DARK_BLUE; bg1.line.color.rgb = DARK_BLUE

tf1 = bg1.text_frame
tf1.margin_left = Inches(1.0); tf1.margin_top = Inches(1.8)
p = tf1.paragraphs[0]
p.text = "SHAP-Driven Feature Discovery in Stellar Spectra"
p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = WHITE

p2 = tf1.add_paragraph()
p2.text = "Module 1: Explainable AI & Exploratory Spectral-Line Alignment in SDSS APOGEE DR17"
p2.font.size = Pt(18); p2.font.color.rgb = ACCENT_ORANGE; p2.space_after = Pt(20)

p3 = tf1.add_paragraph()
p3.text = "Researcher: P.K. Edirisooriya (Student ID: IT23330696)\nProject ID: J26-IT-449 | IT4010 Research Project — Progress Presentation 1 (PP1)\nSLIIT School of Computing"
p3.font.size = Pt(14); p3.font.color.rgb = WHITE

speaker_notes.append("Slide 1 (Title): Good morning/afternoon respected examiners. I am P.K. Edirisooriya (IT23330696). Today I present Progress Presentation 1 for Module 1 of Project J26-IT-449: 'SHAP-Driven Feature Discovery in Stellar Spectra', focusing on Explainable AI and exploratory spectral-line alignment in APOGEE DR17.")

# ── Slide 2: Research Problem & Gap ───────────────────────────────────────────
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "1. Research Problem & Literature Gap")
add_content_box(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "The Astrophysical Problem", [
    "Wide-field spectroscopic surveys capture millions of stellar spectra.",
    "Deep neural networks achieve high classification accuracy on spectra.",
    "Crucial Limitation: Deep neural networks operate as black boxes without physical justification.",
    "Astronomers cannot verify if predictions stem from genuine physical absorption lines or survey artifacts.",
    "Uncertain physical grounding hinders scientific adoption in astrophysics."
])
add_content_box(s2, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "The Evaluated Research Gap", [
    "Most automated spectral classification papers report only accuracy without explainability.",
    "Existing XAI studies in astronomy rely on qualitative visual plots without quantitative validation.",
    "Absence of baseline-controlled matching against NIST laboratory atomic transitions.",
    "Lack of objective Faithfulness (AOPC) and Stability metrics under survey noise.",
    "Our Contribution: A baseline-controlled XAI evaluation framework testing physical alignment and algorithmic faithfulness."
])
speaker_notes.append("Slide 2 (Problem & Gap): While machine learning achieves high accuracy in stellar classification, deep networks operate as black boxes. In astronomy, predictions without physical justification cannot be trusted. Our research evaluates whether Gradient SHAP explanations align with known stellar physics and remain faithful to model decisions.")

# ── Slide 3: Individual Objective & Scope ─────────────────────────────────────
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "2. Research Objectives & Component Scope")
add_content_box(s3, Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.2), "Individual Component Scope (Module 1)", [
    "Primary Objective: Develop a Gradient SHAP-based interpretable stellar spectroscopy framework identifying influential wavelength regions and evaluating their alignment with NIST atomic lines.",
    "Specific Objective 1: Curate and standardize 1,000 SDSS-IV APOGEE DR17 near-infrared spectra into 4,000-channel arrays (processed cohort supplied; raw FITS archived separately).",
    "Specific Objective 2: Benchmark 1D-CNN-BiLSTM against 1D-ResNet, Standard CNN, and classical baselines on a zero-leakage split.",
    "Specific Objective 3: Formulate the Spectral Line Alignment Score and benchmark against a 1,000-trial Monte Carlo random baseline.",
    "Specific Objective 4: Quantify XAI Faithfulness (AOPC) and Stability (Jaccard similarity under simulated survey noise).",
    "Specific Objective 5: Deploy an interactive 6-page Streamlit research application with genuine per-upload Gradient SHAP inference."
])
speaker_notes.append("Slide 3 (Objectives): My individual objective is to establish a mathematically sound, physically evaluated XAI framework for APOGEE spectra. I have delivered on all 5 specific objectives for this PP1 milestone.")

# ── Slide 4: Real Astronomical Dataset ────────────────────────────────────────
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "3. Observational Dataset: SDSS-IV APOGEE DR17")
add_content_box(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Survey & Cohort Breakdown", [
    "Survey: SDSS-IV APOGEE-2 DR17 (Apache Point Observatory 2.5m).",
    "Spectrograph: Cryogenic near-infrared, R ~ 22,500.",
    "Passband: H-band (15,100 to 17,000 Å in rest-frame vacuum).",
    "Catalog Manifest (N = 1,004): K: 649 (64.6%), G: 253 (25.2%), F: 61 (6.1%), A: 25 (2.5%), B: 16 (1.6%).",
    "Active Processed Cohort (N = 1,000): K: 646 (64.6%), G: 252 (25.2%), F: 61 (6.1%), A: 25 (2.5%), B: 16 (1.6%).",
    "Partition Allocation: Train = 700, Validation = 150, Test = 150."
])
add_content_box(s4, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Zero-Leakage Split Class Counts", [
    "Train Set (N = 700): K: 452 (64.6%), G: 176 (25.1%), F: 43 (6.1%), A: 17 (2.4%), B: 12 (1.7%).",
    "Validation Set (N = 150): K: 97 (64.7%), G: 38 (25.3%), F: 9 (6.0%), A: 4 (2.7%), B: 2 (1.3%).",
    "Test Set (N = 150): K: 97 (64.7%), G: 38 (25.3%), F: 9 (6.0%), A: 4 (2.7%), B: 2 (1.3%).",
    "Pairwise Disjointness: 0 shared 2MASS object IDs across all three splits.",
    "Scientific Disclosure: Labels are effective-temperature proxy classes; infrared surveys naturally sample predominantly cool Galactic stars."
])
speaker_notes.append("Slide 4 (Dataset): Our dataset contains 1,004 catalog entries and 1,000 verified processed APOGEE DR17 spectra. The clean zero-leakage split allocates 700 stars to training, 150 to validation, and 150 to testing with exactly 0 shared 2MASS object IDs. The test set comprises 97 K, 38 G, 9 F, 4 A, and 2 B stars, mirroring the survey's natural red giant dominance.")

# ── Slide 5: Spectral Preprocessing Pipeline ──────────────────────────────────
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "4. Preprocessing & Spectral Representation")
add_content_box(s5, Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.2), "Standardized Preprocessing Engine", [
    "Step 1: Vacuum Wavelength Extraction: log10(lambda) = CRVAL1 + CDELT1 * (pixel - (CRPIX1 - 1)) covering 15,100.8 to 16,999.8 Å.",
    "Step 2: Masking & Filtering: Non-positive flux and bad detector pixels masked and interpolated.",
    "Step 3: Uniform Log-Grid Resampling: Raw channels resampled to 4,000 uniform logarithmic points.",
    "Step 4: Spectral Dispersion: Varies smoothly across the band (~0.475 Å/pixel), preserving multi-pixel absorption profiles.",
    "Step 5: Running Median Continuum Normalization: 101-pixel running median filter divides out pseudo-continuum curvature.",
    "Final Representation: Uniform (N, 4000) float32 arrays with continuum centered near 1.0."
])
speaker_notes.append("Slide 5 (Preprocessing): We resample spectra to 4,000 logarithmic channels across the APOGEE H-band, followed by a 101-pixel running median continuum normalization that matches both training and application inference.")

# ── Slide 6: Research Rigor — Leakage Audit ───────────────────────────────────
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "5. Research Integrity: Data Leakage Audit & Fix")
add_content_box(s6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Audit Discovery (Historical Split)", [
    "A nested indexing bug in legacy 01_data_preprocess.py assigned sub-split indices to master array masks.",
    "Impact: 102 training stars were duplicated into the historical 150-sample test set.",
    "Mathematical Consequence: Historical model achieved an inflated 91.33% accuracy due to data leakage.",
    "Audited & Documented: Historical weights and metrics preserved as historical evidence for transparency."
])
add_content_box(s6, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Corrected Zero-Leakage Benchmark", [
    "Partitioned dataset into strictly disjoint astronomical subsets.",
    "Train Set: 700 independent stars.",
    "Validation Set: 150 independent stars.",
    "Test Set: 150 independent stars.",
    "Disjointness Verification: Exactly 0 shared 2MASS object identifiers (objid) across all partitions.",
    "All reported benchmarks independently recomputed on this verified clean split."
])
speaker_notes.append("Slide 6 (Leakage Audit): Research integrity is paramount. Our technical audit identified an indexing bug in the historical split where 102 stars overlapped. We fixed this with a verified zero-leakage partition of 700/150/150 independent stars, confirmed by 0 shared 2MASS object IDs.")

# ── Slide 7: Model Architectures ──────────────────────────────────────────────
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "6. Deep Learning Architectures Evaluated")
add_content_box(s7, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "1D-CNN-BiLSTM (Proposed)", [
    "Hybrid Spectral Architecture",
    "Conv1: 1→32 (k=11, s=4, p=5), BN, ReLU, MaxPool(2) → (B, 32, 500)",
    "Conv2: 32→64 (k=7, s=4, p=3), BN, ReLU, MaxPool(2) → (B, 64, 62)",
    "BiLSTM: 1 layer, input=64, hidden=64/direction → (B, 62, 128)",
    "Temporal Pooling: Global max pool over 62 steps → (B, 128)",
    "Head: Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→5)"
])
add_content_box(s7, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "1D-ResNet", [
    "Deep Residual Baseline",
    "Stem Conv: 1→32 (k=7, s=2, p=3), BN, ReLU, MaxPool(2)",
    "3 Residual Blocks: Conv1d with kernel=3 and skip projections",
    "Channels: 32 → 64 → 128",
    "AdaptiveAvgPool1d(1) → Linear(128→5)",
    "Tests deep residual representations on 1D spectral data"
])
add_content_box(s7, Inches(8.8), Inches(1.5), Inches(3.6), Inches(5.2), "Standard 1D-CNN", [
    "Multi-Layer Conv Baseline",
    "Conv1: 1→32 (kernel 15, padding 7)",
    "Conv2: 32→64 (kernel 7, padding 3)",
    "Conv3: 64→128 (kernel 5, padding 2)",
    "MaxPool1D downsampling after each block",
    "Flatten → Linear(128→64) → Dropout(0.3) → Linear(64→5)"
])
speaker_notes.append("Slide 7 (Architectures): Here is the exact architecture from model_architectures.py. Our proposed CNN-BiLSTM uses two convolutional blocks with kernels 11 and 7, followed by a 1-layer bidirectional LSTM with 64 units per direction, global temporal max pooling, and a dense classification head without attention.")

# ── Slide 8: Model Evaluation Benchmark ───────────────────────────────────────
s8 = prs.slides.add_slide(blank_layout)
add_header(s8, "7. Model Benchmark on Zero-Leakage Test Set")
add_content_box(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Recalculated Benchmark (N = 150 Clean Test)", [
    "1D-CNN-BiLSTM (Ours): Acc = 77.33% | Bal-Acc = 74.15% | Macro-F1 = 0.6406",
    "Logistic Regression (L2): Acc = 84.67% | Bal-Acc = 52.39% | Macro-F1 = 0.5368",
    "Random Forest (100 Trees): Acc = 84.00% | Bal-Acc = 56.87% | Macro-F1 = 0.5758",
    "Majority Zero-Rule: Acc = 64.67% | Bal-Acc = 20.00% | Macro-F1 = 0.1571",
    "1D-ResNet: Acc = 46.00% | Bal-Acc = 42.69% | Macro-F1 = 0.3116",
    "Standard 1D-CNN: Acc = 32.00% | Bal-Acc = 29.17% | Macro-F1 = 0.1699"
])
add_content_box(s8, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Class Breakdown & Balanced Superiority", [
    "Empirical Test Support & F1: K: 78/97 (F1=0.8715), G: 28/38 (F1=0.6667), F: 6/9 (F1=0.5217), A: 2/4 (F1=0.5714), B: 2/2 (F1=0.5714).",
    "Confusion Matrix: [[2,2,0,0,0], [0,2,0,0,0], [0,1,6,2,0], [0,0,6,28,4], [1,0,2,16,78]].",
    "Nominal Accuracy Trap: Classical models reach ~84% accuracy by predicting majority K stars, yielding poor balanced accuracy (52–57%).",
    "CNN-BiLSTM Superiority: Delivers 74.15% Balanced Accuracy and 0.6406 Macro-F1, outperforming all baselines on minority classes.",
    "Minority Note: 100% recall on 2 B-type stars has a wide Wilson CI [34.2%, 100%] and must not be over-generalized."
])
speaker_notes.append("Slide 8 (Benchmark): On the clean 150-star test set, CNN-BiLSTM achieves 77.33% accuracy and 74.15% balanced accuracy. While Logistic Regression (84.67%) and Random Forest (84.00%) have higher nominal accuracy, they achieve this by predicting majority K stars, collapsing balanced accuracy to 52.4% and 56.9%. Our model delivers substantially higher balanced accuracy across all 5 classes (A: 0.57, B: 0.57, F: 0.52, G: 0.67, K: 0.87).")

# ── Slide 9: Gradient SHAP Explainability ─────────────────────────────────────
s9 = prs.slides.add_slide(blank_layout)
add_header(s9, "8. Explainable AI: Gradient SHAP Framework")
add_content_box(s9, Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.2), "GradientExplainer Formulation & Implementation", [
    "Method: shap.GradientExplainer (Gradient SHAP) applied to the PyTorch CNN-BiLSTM architecture.",
    "Methodological Choice: PyTorch recurrent nn.LSTM modules are incompatible with DeepLIFT graph tracing; Gradient SHAP approximates Shapley values via expected gradients.",
    "Background Reference: 30 real stellar spectra drawn from the training partition to model natural continuum variance.",
    "Clean Attribution Tensor: (150, 4000, 5) float32 array — per-star, per-wavelength, per-class importance.",
    "Absolute Average Attribution Score (AAAS): Quantifies feature salience across the 4,000 wavelength channels.",
    "Numerical Integrity: All saved attribution values are strictly finite (0 NaN, 0 Inf)."
])
speaker_notes.append("Slide 9 (Gradient SHAP): We use shap.GradientExplainer — Gradient SHAP — because recurrent LSTM layers are incompatible with DeepLIFT graph tracing. Gradient SHAP computes expected gradients over 30 real training spectra, producing a clean (150, 4000, 5) attribution tensor with zero non-finite values.")

# ── Slide 10: Scientific Line Matching ────────────────────────────────────────
s10 = prs.slides.add_slide(blank_layout)
add_header(s10, "9. Scientific Atomic-Line Validation")
add_content_box(s10, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Peak Detection & Matching Setup", [
    "Compiled 31 NIR literature atomic lines from NIST and APOGEE survey references (H Brackett, Fe I, Si I, Al I, Mg I, C I, N I, O I).",
    "Detected 57 salient attribution peaks using 85th percentile prominence threshold (minimum distance 15 bins).",
    "Matching window: ±3.5 Å tolerance in rest-frame vacuum wavelength.",
    "Total Matched Peaks: 6 out of 57 detected peaks matched catalog lines."
])
add_content_box(s10, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Verified Matched Stellar Lines", [
    "1. O I (15,870.8 Å): Attribution strength = 0.1338 (Strongest peak)",
    "2. H Brackett-14 (15,263.1 Å): Attribution strength = 0.0309",
    "3. N I (16,767.3 Å): Attribution strength = 0.0215",
    "4. Si I (15,959.0 Å): Attribution strength = 0.0179",
    "5. Mg I (15,742.5 Å): Attribution strength = 0.0173",
    "6. O I (15,570.0 Å): Attribution strength = 0.0172",
    "Observed Alignment Score: 10.53% (6 / 57 peaks matched)."
])
speaker_notes.append("Slide 10 (Line Matching): We matched detected attribution peaks against 31 NIST literature lines within 3.5 Å. The 6 verified matches are O I, H Brackett-14, N I, Si I, Mg I, and a second O I feature, yielding an observed alignment score of 10.53%.")

# ── Slide 11: Monte Carlo Random Baseline ─────────────────────────────────────
s11 = prs.slides.add_slide(blank_layout)
add_header(s11, "10. Monte Carlo Statistical Significance Test")
add_content_box(s11, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Experimental Methodology", [
    "Scientific Question: Does 10.53% line alignment exceed random chance?",
    "Simulated 1,000 Monte Carlo trials drawing 57 random peak wavelengths across the 15,100–17,000 Å passband.",
    "Evaluated each trial under identical 3.5 Å matching tolerance.",
    "Computed the empirical null distribution of random alignment scores."
])
add_content_box(s11, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Statistical Result & Honest Qualification", [
    "Random Expectation: 10.19% ± 3.94%",
    "Observed Model Score: 10.53% (6 matches)",
    "Empirical p-value: p = 0.5380 (538 of 1,000 random trials matched >= 6 lines)",
    "Statistical Finding: Alignment is not statistically significant above this simple uniform null (p > 0.05).",
    "Scientific Conclusion: The observed narrow atomic-line alignment did not significantly exceed random baseline; additional analysis is required to evaluate broader molecular band or continuum reliance."
])
speaker_notes.append("Slide 11 (Monte Carlo): To evaluate whether 10.53% alignment is meaningful, we ran a 1,000-trial Monte Carlo test. The random baseline mean is 10.19% (p=0.538), showing that narrow atomic-line alignment is not statistically significant above chance. We report this finding candidly rather than overclaiming physical line discovery.")

# ── Slide 12: XAI Faithfulness & Stability ────────────────────────────────────
s12 = prs.slides.add_slide(blank_layout)
add_header(s12, "11. Objective XAI Faithfulness & Stability")
add_content_box(s12, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Faithfulness: AOPC Metric", [
    "Area Over Perturbation Curve (AOPC) evaluated on 50 correctly classified test stars.",
    "Sequentially masked top-k SHAP channels vs random channels (k up to 1,200 bins).",
    "Top-k AOPC: 0.7507 ± 0.1898 (Rapid confidence drop).",
    "Random AOPC: 0.1192 ± 0.1280 (Minimal degradation).",
    "Faithfulness Advantage: 6.30x advantage over random masking under this controlled perturbation design."
])
add_content_box(s12, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Stability: Jaccard Overlap under Noise", [
    "Evaluated attribution consistency under simulated additive Gaussian noise.",
    "High SNR (~100, sigma=0.01): Mean Jaccard overlap = 0.8069 ± 0.0628.",
    "Nominal Survey SNR (~50, sigma=0.02): Mean Jaccard overlap = 0.7768 ± 0.0635.",
    "Low SNR (~25, sigma=0.04): Mean Jaccard overlap = 0.6853 ± 0.0539.",
    "Confirms stability under simulated additive Gaussian noise perturbations."
])
speaker_notes.append("Slide 12 (Faithfulness & Stability): We objectively validated explanation quality. Top-k SHAP masking degrades model confidence 6.30x faster than random perturbations (AOPC=0.75 vs 0.12), and explanations maintain over 77.6% Jaccard overlap under simulated survey noise.")

# ── Slide 13: Interactive Streamlit Web UI ────────────────────────────────────
s13 = prs.slides.add_slide(blank_layout)
add_header(s13, "12. Interactive Streamlit Web Application")
add_content_box(s13, Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.2), "Operational Research Dashboard (app.py)", [
    "Page 1: 📊 Dashboard — Dataset manifest distribution (N=1,004), verified KPIs (77.33% Acc, 74.15% Bal-Acc), and 150-sample confusion matrix.",
    "Page 2: 🔬 Upload & Analyse — Live upload of FITS, CSV, NPZ, or NPY spectra; WCS 1-indexed wavelength parsing; training-parity continuum normalization; genuine live per-upload Gradient SHAP attribution (<1.5s latency).",
    "Page 3: 📈 Model Comparison — Clean split benchmark comparing 1D-CNN-BiLSTM vs ResNet, CNN, and classical baselines (RF, LR).",
    "Page 4: ⚛️ Scientific XAI Validation — Interactive tabs for NIST Line Alignment (p=0.538), AOPC Faithfulness (6.30x), Jaccard Stability, and Monte Carlo curves.",
    "Page 5: 🧪 Test Evaluation — Confusion matrix, automated assertion checks, and per-class precision/recall for all 5 classes.",
    "Page 6: 📚 About & Limitations — Full scientific disclosures on proxy labeling, class imbalance, and reproducibility commands."
])
speaker_notes.append("Slide 13 (Application): Our 6-page Streamlit web dashboard supports live upload of FITS, CSV, NPZ, and NPY spectra with training-parity preprocessing and computes real-time Gradient SHAP attributions specifically for uploaded spectra.")

# ── Slide 14: Quality Assurance & Testing ─────────────────────────────────────
s14 = prs.slides.add_slide(blank_layout)
add_header(s14, "13. Automated Quality Assurance & Verification")
add_content_box(s14, Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.2), "69/69 Passing Automated Assertions (pytest 9.1.1)", [
    "TestZeroLeakageDataset (4 tests): Verified 700/150/150 counts, 4000 feature length, exactly 0 sample overlap, all finite values.",
    "TestModelArchitectures (4 tests): Verified forward pass dimensions (B, 5), absence of NaNs, and clean checkpoint loading.",
    "TestDirectRecomputations (3 tests): Verified 77.33% accuracy recomputed directly from model weights on test spectra, pairwise disjoint 2MASS IDs, and clean SHAP tensor shape.",
    "TestScientificLineValidation (3 tests): Verified positive peak detection, alignment score bounds, and Monte Carlo baseline existence.",
    "TestXAIEvaluation (2 tests): Verified positive AOPC score, >1.0x faithfulness advantage, and >=0.70 Jaccard stability at SNR 100.",
    "TestBaselineBenchmarks (2 tests): Verified baseline result existence and CNN-BiLSTM balanced accuracy superiority over classical models.",
    "TestCheckpointDeepCopyAndFormatContracts (3 tests): Deep-copy snapshot isolation test, 4000-bin format contracts, real FITS inference.",
    "TestUploadPreprocessingContracts (6 tests): Rejection of NaN/Inf, length validation, passband overlap checks, pre-normalization validation, and raw count normalization.",
    "TestManifest & Pipeline (42 tests): Verified 100% manifest column integrity, TEFF boundary definitions, and confusion matrix consistency.",
    "100% Pass Rate: 69 passed in 3.44 seconds."
])
speaker_notes.append("Slide 14 (Testing): We implemented 69 automated pytest assertions covering dataset zero-leakage, architecture dimensions, direct recomputation from model weights, XAI faithfulness, scientific line alignment, and strict upload preprocessing contracts. All 69 tests pass with a 100% success rate.")

# ── Slide 15: Conclusion & Future Work ────────────────────────────────────────
s15 = prs.slides.add_slide(blank_layout)
add_header(s15, "14. Conclusion & Post-PP1 Research Roadmap")
add_content_box(s15, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "PP1 Milestone Deliverables", [
    "1,000 verified APOGEE DR17 spectra in clean zero-leakage split.",
    "Historical 102-sample overlap discovered and corrected.",
    "1D-CNN-BiLSTM achieved 77.33% accuracy and 74.15% balanced accuracy.",
    "Formulated Spectral Line Alignment Score (10.53%, p=0.538).",
    "XAI Faithfulness (6.30x AOPC ratio) and Stability (>77.6%) evaluated.",
    "6-page interactive research application with live Gradient SHAP.",
    "69/69 passing automated assertions."
])
add_content_box(s15, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Post-PP1 Roadmap (Towards PP2 & Final)", [
    "Phase 1: Synthetic Spectral Augmentation (Turbospectrum / Kurucz models) to expand minority O, B, and A class representation.",
    "Phase 2: Molecular Bandhead Matching: Expand reference list from atomic lines to near-infrared CO and CN ro-vibrational transitions.",
    "Phase 3: Epistemic Uncertainty Quantification via Monte Carlo Dropout and deep evidential learning.",
    "Phase 4: Comparative XAI benchmarking against Integrated Gradients and DeepLIFT."
])
speaker_notes.append("Slide 15 (Conclusion): In conclusion, Module 1 is fully implemented, verified, and audit-compliant for PP1. Our next steps toward PP2 include synthetic augmentation for hot stars and molecular bandhead analysis. Thank you. I welcome your questions.")

# Save presentation
out_pptx = ROOT / "presentation" / "STELLAR_SPECTRA_PP1_PRESENTATION_CORRECTED.pptx"
prs.save(str(out_pptx))
print(f"Saved: {out_pptx}")

# Save speaker notes
doc_notes = docx.Document()
doc_notes.add_heading("PP1 PRESENTATION SPEAKER NOTES & TIMED SCRIPT (CORRECTED)", level=1)
for i, note in enumerate(speaker_notes, 1):
    doc_notes.add_heading(f"Slide {i}", level=2)
    doc_notes.add_paragraph(note)
out_docx = ROOT / "presentation" / "PRESENTATION_SPEAKER_NOTES_CORRECTED.docx"
doc_notes.save(str(out_docx))
print(f"Saved: {out_docx}")
