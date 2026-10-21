"""
generate_viva_full.py
=====================
Generates a FULL comprehensive bilingual Viva / PP1 Panel Preparation Guide
covering: System pages, Previous problems faced, All expected panel questions,
Sinhala explanations throughout.

Run:  python generate_viva_full.py
Out:  reports/FULL_VIVA_PANEL_GUIDE.pdf
"""

import io, os, time
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)

pdfmetrics.registerFont(TTFont('Nirmala',  'C:/Windows/Fonts/Nirmala.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('NirmalaB', 'C:/Windows/Fonts/Nirmala.ttc', subfontIndex=1))

ROOT        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT, 'reports', 'FULL_VIVA_PANEL_GUIDE.pdf')

# ── Colours ───────────────────────────────────────────────────────────────────
CN   = colors.HexColor('#0F2043')   # navy
CB   = colors.HexColor('#1565C0')   # blue
CG   = colors.HexColor('#2E7D32')   # green
CGB  = colors.HexColor('#E8F5E9')   # green bg
CR   = colors.HexColor('#B71C1C')   # red
CRB  = colors.HexColor('#FFEBEE')   # red bg
CA   = colors.HexColor('#E65100')   # amber
CAB  = colors.HexColor('#FFF3E0')   # amber bg
CP   = colors.HexColor('#4A148C')   # purple
CPB  = colors.HexColor('#F3E5F5')   # purple bg
CLG  = colors.HexColor('#F8F9FA')   # light grey
CBD  = colors.HexColor('#D0D7DE')   # border
CW   = colors.white

W = 515   # usable page width pts


# ── Style factory ─────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

EH  = S('EH',  fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=CB, spaceBefore=7, spaceAfter=3)
EB  = S('EB',  fontName='Helvetica',      fontSize=9,  leading=13, textColor=colors.HexColor('#1A1A1A'))
EBB = S('EBB', fontName='Helvetica-Bold', fontSize=9,  leading=13, textColor=colors.HexColor('#1A1A1A'))
SB  = S('SB',  fontName='Nirmala',        fontSize=10, leading=17, textColor=colors.HexColor('#0D3B00'))
SBB = S('SBB', fontName='NirmalaB',       fontSize=10, leading=17, textColor=CG)
VQ  = S('VQ',  fontName='Helvetica-Bold', fontSize=9.5,leading=14, textColor=CA, spaceBefore=3)
VA  = S('VA',  fontName='Helvetica',      fontSize=9,  leading=13, textColor=colors.HexColor('#1A1A1A'))
TIT = S('TIT', fontName='Helvetica-Bold', fontSize=22, leading=28, textColor=CN, spaceAfter=4, alignment=1)
SUB = S('SUB', fontName='Helvetica',      fontSize=10, leading=14, textColor=colors.HexColor('#555'), alignment=1)
SIN_TIT = S('ST', fontName='NirmalaB', fontSize=13, leading=20, textColor=CG, alignment=1)
FT  = S('FT',  fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#555'), alignment=1)
WARN = S('WARN', fontName='Helvetica-Bold', fontSize=9.5, leading=14, textColor=CR)


def banner(en, sin, bg=CN):
    data = [[Paragraph(f"<font color='white'><b>{en}</b></font>",
                       S('bh', fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=CW)),
             Paragraph(f"<font color='#AAFFAA'>{sin}</font>",
                       S('bs', fontName='Nirmala', fontSize=10, leading=14, textColor=colors.HexColor('#AAFFAA')))]]
    t = Table(data, colWidths=[355, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), bg),
        ('TOPPADDING', (0,0),(-1,-1), 8), ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('LEFTPADDING', (0,0),(-1,-1), 10), ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


def bi(en, sin, lw=258):
    """Bilingual side-by-side block."""
    rw = W - lw
    data = [[Paragraph(en, EB), Paragraph(sin, SB)]]
    t = Table(data, colWidths=[lw, rw])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(0,-1), colors.HexColor('#EEF4FF')),
        ('BACKGROUND', (1,0),(1,-1), CGB),
        ('BOX', (0,0),(-1,-1), 0.5, CBd()), ('INNERGRID', (0,0),(-1,-1), 0.5, CBd()),
        ('TOPPADDING', (0,0),(-1,-1), 5), ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 6), ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
    ]))
    return t


def CBd(): return CBD


def qa(q, en, sin, qbg=CAB, qcolor=CA):
    """Q&A block with bilingual answer."""
    data = [
        [Paragraph(f"❓  {q}", S('qq', fontName='Helvetica-Bold', fontSize=9.5, leading=14, textColor=qcolor)), ''],
        [Paragraph(f"<b>Answer:</b> {en}", VA), Paragraph(f"<b>පිළිතුර:</b> {sin}", SB)],
    ]
    t = Table(data, colWidths=[258, 257])
    t.setStyle(TableStyle([
        ('SPAN', (0,0),(1,0)),
        ('BACKGROUND', (0,0),(1,0), qbg),
        ('BACKGROUND', (0,1),(0,1), colors.HexColor('#EEF4FF')),
        ('BACKGROUND', (1,1),(1,1), CGB),
        ('BOX', (0,0),(-1,-1), 1, qcolor),
        ('INNERGRID', (0,0),(-1,-1), 0.5, CBD),
        ('TOPPADDING', (0,0),(-1,-1), 5), ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 6), ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
    ]))
    return t


def problem_box(title, problem_en, problem_sin, fix_en, fix_sin):
    """Red problem + green fix box."""
    data = [
        [Paragraph(f"⚠  PROBLEM: {title}", S('pt', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=CW)), ''],
        [Paragraph(f"<b>What happened:</b> {problem_en}", EB),
         Paragraph(f"<b>සිදු වූ දේ:</b> {problem_sin}", SB)],
        [Paragraph(f"✅ <b>How fixed:</b> {fix_en}", VA),
         Paragraph(f"✅ <b>නිරාකරණය:</b> {fix_sin}", SB)],
    ]
    t = Table(data, colWidths=[258, 257])
    t.setStyle(TableStyle([
        ('SPAN', (0,0),(1,0)),
        ('BACKGROUND', (0,0),(1,0), CR),
        ('BACKGROUND', (0,1),(0,1), CRB),
        ('BACKGROUND', (1,1),(1,1), colors.HexColor('#FFF9E6')),
        ('BACKGROUND', (0,2),(0,2), CGB),
        ('BACKGROUND', (1,2),(1,2), CGB),
        ('BOX', (0,0),(-1,-1), 1, CR),
        ('INNERGRID', (0,0),(-1,-1), 0.5, CBD),
        ('TOPPADDING', (0,0),(-1,-1), 5), ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 6), ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
    ]))
    return t


def info(en, sin, bg=CB):
    data = [[Paragraph(
        f"<font color='white'><b>{en}</b></font><br/><font color='#DDFFDD'>{sin}</font>",
        S('ib', fontName='Nirmala', fontSize=9.5, leading=15, textColor=CW))]]
    t = Table(data, colWidths=[W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), bg),
        ('TOPPADDING', (0,0),(-1,-1), 6), ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING', (0,0),(-1,-1), 10), ('RIGHTPADDING', (0,0),(-1,-1), 10),
    ]))
    return t


def sp(n=0.2): return Spacer(1, n*cm)
def hr(): return HRFlowable(width="100%", thickness=0.8, color=CBd(), spaceAfter=4)


# ═════════════════════════════════════════════════════════════════════════════
def build():
    story = []

    # ── COVER ─────────────────────────────────────────────────────────────────
    story += [sp(2),
        Paragraph("FULL PP1 VIVA PANEL PREPARATION GUIDE", TIT),
        Paragraph("Stellar Spectra XAI Research System — J26-IT-449", SUB),
        sp(0.3),
        Paragraph("සම්පූර්ණ PP1 Viva Panel සූදානම් වීමේ මාර්ගෝපදේශය", SIN_TIT),
        Paragraph("Previous Problems · All Panel Questions · Sinhala Explanations · Page Walkthroughs",
                  S('s2', fontName='Nirmala', fontSize=10, leading=15,
                    textColor=colors.HexColor('#555'), alignment=1)),
        sp(0.5),
        HRFlowable(width="100%", thickness=2.5, color=CN, spaceAfter=10),
    ]

    cover_rows = [
        ["Student",       "P.K. Edirisooriya  |  IT23330696"],
        ["Project ID",    "J26-IT-449  |  SLIIT IT4010 Research Project — PP1 Assessment"],
        ["Module",        "Module 1: SHAP-Driven Feature Discovery in Stellar Spectra"],
        ["Model",         "1D-CNN-BiLSTM  |  Test Acc: 77.33%  |  Bal-Acc: 74.15%  |  Macro F1: 0.6406"],
        ["Test Split",    "Zero-leakage  |  700 Train / 150 Val / 150 Test"],
        ["XAI Method",    "Gradient SHAP (shap.GradientExplainer)"],
        ["Validated By",  "AOPC Faithfulness (0.7507 vs 0.1192)  |  Jaccard Stability (0.8069)"],
        ["Generated",     time.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    t = Table([[Paragraph(f"<b>{k}</b>", EBB), Paragraph(v, EB)] for k,v in cover_rows],
              colWidths=[100, 415])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), CLG), ('BOX',(0,0),(-1,-1),0.5,CBD),
        ('INNERGRID',(0,0),(-1,-1),0.5,CBD),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',(0,0),(-1,-1),7),
    ]))
    story += [t, sp(0.3),
        info("This guide covers EVERY question a PP1 panel may ask — including the problems you faced and how you fixed them.",
             "මෙම මාර්ගෝපදේශය PP1 panel ඇසිය හැකි සෑම ප්‍රශ්නයක්ම ආවරණය කරයි — ඔබ මුහුණ දුන් ගැටලු සහ ඒවා නිරාකරණය කළ ආකාරය ඇතුළුව."),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 1 — PROBLEMS FACED & HOW THEY WERE FIXED
    # ═══════════════════════════════════════════════════════════════════════════
    story += [
        banner("CHAPTER 1 — PROBLEMS FACED & HOW THEY WERE FIXED",
               "ගැටලු සහ ඒවා නිරාකරණය කළ ආකාරය", CR),
        sp(0.2),
        info("Panel members WILL ask about these. Know them cold!",
             "Panel members මෙම ගැටලු ගැන අනිවාර්යෙන්ම අසයි. හොඳින් ඉගෙන ගන්න!", CR),
        sp(0.3),
    ]

    # Problem 1 — Data Leakage
    story += [
        problem_box(
            "Data Leakage — 91.33% Accuracy (Historical Bug)",
            "The original train/val/test split had a nested indexing bug in 01_data_preprocess.py. "
            "Val and test indices were re-indexed from 0 after the train split, causing test samples "
            "to overlap with training data. The model appeared to achieve 91.33% accuracy, but this "
            "was artificially inflated — the model had seen test data during training.",
            "01_data_preprocess.py හි nested indexing bug නිසා val/test indices train split ලෙස 0 ලෙස re-index වීය. "
            "Test samples training data සමඟ overlap විය. Model 91.33% accuracy ලෙස පෙනිණ — "
            "නමුත් model test data training ලෙස දැක ඇති නිසා inflate කළ ප්‍රතිඵලයකි.",
            "We identified the bug using index-overlap analysis. We re-ran the entire pipeline with a "
            "corrected zero-leakage split (700/150/150) producing the honest benchmark: "
            "77.33% accuracy, 74.15% balanced accuracy, Macro F1 0.6406. "
            "The old results are preserved as 'historical artifact' in metrics.json.",
            "Index-overlap analysis ලෙස bug හඳුනා ගත්තෙමු. Corrected zero-leakage split (700/150/150) ලෙස "
            "pipeline සම්පූර්ණයෙන් re-run කළෙමු. Honest benchmark: 77.33% accuracy, 74.15% balanced accuracy, "
            "Macro F1 0.6406. පැරණි results metrics.json හි 'historical artifact' ලෙස preserve කළෙමු."
        ),
        sp(0.2),
        qa("Why was the original accuracy 91.33% and now it is 77.33%?",
           "The original split had a data leakage bug. Val/test indices were re-indexed after the train "
           "split, causing test samples to overlap with training data, inflating accuracy. After correcting "
           "the split to a genuine zero-leakage 700/150/150 partition, the honest test accuracy is 77.33%.",
           "Original split හි data leakage bug ඇති විය. Val/test indices train split ලෙස re-index වීම "
           "test samples training data ලෙස overlap කිරීමට හේතු විය — accuracy inflate විය. "
           "Genuine zero-leakage 700/150/150 partition ලෙස correct කළ පසු honest test accuracy 77.33% ලැබිණ.",
           qbg=CRB, qcolor=CR),
        sp(0.3),
    ]

    # Problem 2 — SHAP Shape Mismatch
    story += [
        problem_box(
            "SHAP Tensor Shape Mismatch (150×4000×5)",
            "The saved SHAP attributions file (shap_attributions_clean.npz) has shape (150, 4000, 5) — "
            "150 test samples, 4000 wavelength channels, 5 classes. The original app.py code tried to "
            "render this as a 2D array, causing crashes and incorrect SHAP visualisations.",
            "Saved SHAP attributions (shap_attributions_clean.npz) shape (150, 4000, 5) — "
            "test samples 150, wavelength channels 4000, classes 5. Original app.py code 2D array "
            "ලෙස render කිරීමට attempt කළ නිසා crashes සහ incorrect SHAP visualisations ඇති විය.",
            "Fixed by computing the global mean: np.mean(np.abs(shap_tensor), axis=(0,2)) "
            "→ shape (4000,) → one clean attribution curve. Per-class overlays added as optional traces.",
            "Global mean compute: np.mean(np.abs(shap_tensor), axis=(0,2)) → shape (4000,) → "
            "clean attribution curve. Per-class overlays optional traces ලෙස add කළෙමු."
        ),
        sp(0.3),
    ]

    # Problem 3 — Hardcoded Paths
    story += [
        problem_box(
            "Hardcoded sys.path & os.chdir() — Portability Broken",
            "Training and analysis scripts (train_corrected_split.py, 03–07 scripts) had hardcoded "
            "absolute paths like sys.path.insert(0, r'C:\\Users\\pasan\\...') and os.chdir(). "
            "These prevented the code from running on any other machine.",
            "Training/analysis scripts (train_corrected_split.py, 03–07) C:\\Users\\pasan\\... "
            "ලෙස hardcoded absolute paths (sys.path.insert, os.chdir) ඇති විය. "
            "වෙනත් machine ලෙස code run කිරීම prevent විය.",
            "Replaced all hardcoded paths with ROOT = pathlib.Path(__file__).resolve().parent "
            "and built all paths relative to ROOT. Added sys.path.insert(0, str(ROOT)) for imports.",
            "Hardcoded paths සියල්ල ROOT = pathlib.Path(__file__).resolve().parent ලෙස replace කළෙමු. "
            "සියලු paths ROOT relative ලෙස build කළෙමු. Imports සඳහා sys.path.insert(0, str(ROOT)) add කළෙමු."
        ),
        sp(0.3),
    ]

    # Problem 4 — App.py metrics loading wrong file
    story += [
        problem_box(
            "Dashboard Showing Wrong Metrics (91.33% leaky result)",
            "The load_metrics() function in app.py was reading from metrics.json which contained "
            "the historical inflated 91.33% result. The Dashboard was displaying these wrong numbers "
            "to users.",
            "app.py හි load_metrics() function metrics.json ලෙස read කළ — historical inflated "
            "91.33% result ඇතුළත් file. Dashboard users ලෙස wrong numbers display කළේය.",
            "Fixed load_metrics() to read from results/corrected_split_benchmark.json['1D-CNN-BiLSTM'] "
            "returning the correct 77.33%, 74.15%, 0.6406 values. The old metrics.json kept as "
            "'historical artifact'. load_historical_metrics() added for backward compatibility.",
            "load_metrics() results/corrected_split_benchmark.json['1D-CNN-BiLSTM'] ලෙස read "
            "කිරීමට fix කළෙමු — correct 77.33%, 74.15%, 0.6406. Old metrics.json 'historical artifact' "
            "ලෙස preserve. Backward compatibility සඳහා load_historical_metrics() add කළෙමු."
        ),
        sp(0.3),
    ]

    # Problem 5 — Upload page crashes on wrong FITS
    story += [
        problem_box(
            "Upload Page Crashed on Non-Spectral Files (HST Images, etc.)",
            "When users uploaded incompatible files (e.g. HST Pillars of Creation FITS image, "
            "random noise files, optical spectra), the app showed a raw Python traceback error "
            "with numpy dtype messages that were meaningless to users.",
            "Users incompatible files (HST Pillars of Creation FITS image, random noise, optical spectra) "
            "upload කළ විට, app raw Python traceback error (numpy dtype messages) show කළේය — "
            "users ලෙස meaningless.",
            "Added intelligent error classification with 6 specific friendly messages: "
            "imaging FITS detected, wrong wavelength range, wrong array shape, empty file, "
            "missing column, and a generic fallback. Each message explains what went wrong "
            "and what to upload instead.",
            "Intelligent error classification: friendly messages 6 ක් — imaging FITS detected, "
            "wrong wavelength range, wrong array shape, empty file, missing column, generic fallback. "
            "සෑම message ලෙස wrong දෙය සහ upload කළ යුතු දෙය explain කරයි."
        ),
        sp(0.2),
        qa("What happens when a user uploads a Hubble image FITS file?",
           "The system detects the numpy structured record array (image metadata table) and shows "
           "a clear error: 'Unsupported File Format — your file appears to be an HST/JWST imaging FITS, "
           "not a 1D stellar spectrum.' It lists what to upload instead and shows sample file hints.",
           "System numpy structured record array (image metadata table) detect කර clear error show කරයි: "
           "'Unsupported File Format — HST/JWST imaging FITS, 1D stellar spectrum නොවේ.' "
           "Upload කළ යුතු දෙය list කර sample file hints show කරයි.",
           qbg=CRB, qcolor=CR),
        sp(0.3),
    ]

    # Problem 6 — Test suite failures
    story += [
        problem_box(
            "Test Suite Had 5 Errors (Legacy Split Paths)",
            "The original test_pipeline.py loaded from the old split paths (processed/train.npz) "
            "and expected the historical metrics.json structure. After the corrected split, "
            "5 tests errored because paths and schema had changed.",
            "Original test_pipeline.py old split paths (processed/train.npz) ලෙස load කර "
            "historical metrics.json structure expect කළේය. Corrected split ලෙස 5 tests errored — "
            "paths සහ schema change වී ඇති නිසා.",
            "Fixed TestDatasetSplits to use processed/corrected_clean/{train,val,test}.npz. "
            "Created corrected_metrics fixture from corrected_split_benchmark.json. "
            "Fixed all 5 failing tests. Suite expanded from 48 to 70 passing tests.",
            "TestDatasetSplits processed/corrected_clean/{train,val,test}.npz ලෙස fix. "
            "corrected_metrics fixture corrected_split_benchmark.json ලෙස create. "
            "Failing tests 5 fix. Suite 48 → 70 passing tests ලෙස expand කළෙමු."
        ),
        sp(0.3),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 2 — SYSTEM ARCHITECTURE & TECHNOLOGY
    # ═══════════════════════════════════════════════════════════════════════════
    story += [
        banner("CHAPTER 2 — SYSTEM ARCHITECTURE & TECHNOLOGY STACK",
               "පද්ධති Architecture සහ Technology Stack", CN),
        sp(0.3),
    ]

    story += [
        Paragraph("Architecture Overview", EH),
        bi("The system uses a Python-based layered architecture:\n"
           "• Presentation Layer: Streamlit web UI (6 pages)\n"
           "• Application Layer: Image preprocessing, model loading, result formatting\n"
           "• Classification Layer: Frozen 1D-CNN-BiLSTM (PyTorch)\n"
           "• Explainability Layer: Gradient SHAP attributions\n"
           "• Storage Layer: JSON/CSV/NPZ results, model checkpoints, FITS data",
           "Python-based layered architecture:\n"
           "• Presentation Layer: Streamlit web UI (pages 6)\n"
           "• Application Layer: Preprocessing, model loading, result formatting\n"
           "• Classification Layer: Frozen 1D-CNN-BiLSTM (PyTorch)\n"
           "• Explainability Layer: Gradient SHAP attributions\n"
           "• Storage Layer: JSON/CSV/NPZ, model checkpoints, FITS data"),
        sp(0.2),
    ]

    tech_rows = [
        ["Technology", "Purpose (English)", "අරමුණ (සිංහල)"],
        ["Python 3.x",       "Core programming language",                    "ප්‍රධාන programming language"],
        ["Streamlit",        "Web UI framework — 6-page interactive app",     "Web UI — interactive pages 6"],
        ["PyTorch",          "Load frozen CNN-BiLSTM, run inference",         "Frozen model load, inference"],
        ["SHAP",             "GradientExplainer for attribution computation", "Attribution computation"],
        ["NumPy / SciPy",    "Array ops, resampling, median filter",          "Array ops, resampling, filter"],
        ["Astropy",          "FITS file reading, WCS wavelength extraction",  "FITS reading, WCS extraction"],
        ["Plotly",           "Interactive spectrum and SHAP charts",          "Interactive charts"],
        ["ReportLab",        "In-memory scientific PDF report generation",    "Scientific PDF generation"],
        ["Pandas",           "DataFrame for metrics and benchmark tables",    "Metrics/benchmark tables"],
        ["pytest (70 tests)","Automated test suite for correctness audit",    "Automated test suite"],
    ]
    t = Table([[Paragraph(f"<b>{c}</b>" if i==0 else c,
                          EBB if i==0 else (SB if j==2 else EB))
                for j,c in enumerate(row)] for i,row in enumerate(tech_rows)],
              colWidths=[90, 230, 195])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),CN), ('TEXTCOLOR',(0,0),(-1,0),CW),
        ('INNERGRID',(0,0),(-1,-1),0.5,CBD), ('BOX',(0,0),(-1,-1),0.5,CBD),
        ('TOPPADDING',(0,0),(-1,-1),3), ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('LEFTPADDING',(0,0),(-1,-1),5), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[CLG, CW]),
    ]))
    story += [t, sp(0.2)]

    story += [
        qa("What framework did you use for the web interface?",
           "Streamlit — a Python-based rapid web application framework. It converts Python scripts "
           "into interactive web apps without requiring HTML, CSS or JavaScript. We chose it because "
           "it integrates directly with PyTorch and NumPy, making real-time model inference and "
           "SHAP computation straightforward.",
           "Streamlit — Python-based rapid web application framework. HTML, CSS, JavaScript "
           "නොමැතිව Python scripts interactive web apps ලෙස convert කරයි. PyTorch සහ NumPy "
           "directly integrate කරන නිසා real-time inference සහ SHAP computation straightforward — "
           "ඒ නිසා select කළෙමු."),
        sp(0.2),
        qa("What is the CNN-BiLSTM model architecture?",
           "Conv1d(1→32, kernel=11) → Conv1d(32→64, kernel=7) → Bidirectional LSTM(hidden=64 per direction, "
           "128 concatenated) → MaxPool → Linear(128→64) → Linear(64→5) → Softmax. "
           "Total: ~350K parameters. No attention module.",
           "Conv1d(1→32, k=11) → Conv1d(32→64, k=7) → Bidirectional LSTM(hidden=64/direction, "
           "128 concat) → MaxPool → Linear(128→64) → Linear(64→5) → Softmax. "
           "Parameters ~350K. Attention module නොමැත."),
        sp(0.3),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 3 — PAGE BY PAGE WALKTHROUGH
    # ═══════════════════════════════════════════════════════════════════════════
    story += [banner("CHAPTER 3 — PAGE-BY-PAGE SYSTEM WALKTHROUGH",
                     "Page-by-Page System Walkthrough", CB), sp(0.3)]

    # Page 1 Dashboard
    story += [Paragraph("PAGE 1 — 🏠 Dashboard", EH),
        bi("Shows class distribution bar chart (N=1,004 manifest), corrected model metrics "
           "(77.33% / 74.15% / 0.6406), confusion matrix, and XAI method summary.",
           "Class distribution bar chart (N=1,004), corrected metrics (77.33% / 74.15% / 0.6406), "
           "confusion matrix, XAI method summary show කරයි."),
        sp(0.15),
        qa("What does the confusion matrix show?",
           "A table of true class vs predicted class for all 150 test samples. The diagonal shows "
           "correct predictions. Off-diagonal shows mistakes. Class K has most correct predictions "
           "due to dataset dominance.",
           "True class vs predicted class — test samples 150 සඳහා table. Diagonal correct predictions. "
           "Off-diagonal mistakes. Class K dataset dominance නිසා most correct predictions."), sp(0.2)]

    # Page 2 Spectrum Explorer
    story += [Paragraph("PAGE 2 — 🔭 Spectrum Explorer", EH),
        bi("Browse all 150 test spectra by index slider. Shows normalised flux plot, true label, "
           "predicted label + confidence, SHAP attribution with NIST vertical lines, "
           "and astrophysical properties card.",
           "Index slider ලෙස test spectra 150 browse. Normalised flux plot, true label, "
           "predicted label + confidence, NIST vertical lines සහිත SHAP attribution, "
           "astrophysical properties card show කරයි."),
        sp(0.15),
        qa("What do the vertical dashed lines on the SHAP plot mean?",
           "They mark known atomic absorption lines from the NIST Atomic Spectra Database — "
           "O I (15,871 Å), Mg I (15,745 Å), Si I (15,960 Å), H Br14 (15,263 Å) etc. "
           "They visually show where SHAP attribution peaks align with real physics.",
           "NIST Atomic Spectra Database known atomic absorption lines mark — O I (15,871 Å), "
           "Mg I (15,745 Å), Si I (15,960 Å), H Br14 (15,263 Å) ආදිය. "
           "SHAP attribution peaks real physics ලෙස align ස්ථාන visually show කරයි."), sp(0.2)]

    # Page 3 Upload & Analyse
    story += [Paragraph("PAGE 3 — 🔬 Upload & Analyse", EH),
        bi("Upload any APOGEE-format spectrum (FITS/CSV/NPZ/NPY). System preprocesses → runs "
           "1D-CNN-BiLSTM inference → computes real-time Gradient SHAP → shows astrophysical "
           "properties with PRIMARY FINDING highlighted → offers JSON + PDF download.",
           "APOGEE-format spectrum (FITS/CSV/NPZ/NPY) upload. System preprocess → 1D-CNN-BiLSTM "
           "inference → real-time Gradient SHAP → PRIMARY FINDING highlight ලෙස astrophysical "
           "properties → JSON + PDF download offer කරයි."),
        sp(0.15),
        qa("What preprocessing steps run after upload?",
           "1) Read file (FITS WCS extraction / CSV parsing / NPZ key 'X'). "
           "2) Resample to 4,000 log-spaced grid (15,100–17,000 Å). "
           "3) 101-pixel median continuum normalisation. "
           "4) Validate finite values and correct shape.",
           "1) File read (FITS WCS extraction / CSV parse / NPZ key 'X'). "
           "2) 4,000 log-spaced grid (15,100–17,000 Å) ලෙස resample. "
           "3) 101-pixel median continuum normalisation. "
           "4) Finite values + correct shape validate."),
        sp(0.15),
        qa("What does the 'Primary Astronomical Finding' green box show?",
           "It shows the single most important scientific interpretation of the result for that "
           "spectral class — e.g. for Class G: 'Solar Twin — stable yellow dwarf identical to our Sun.' "
           "Below it is the Astrophysical Interpretation — a one-sentence scientific explanation. "
           "This was added so panel members can immediately understand the result.",
           "'Primary Astronomical Finding' green box, predict කළ spectral class ලෙස "
           "single most important scientific interpretation show — e.g. Class G: "
           "'Solar Twin — stable yellow dwarf identical to our Sun.' "
           "Astrophysical Interpretation — one-sentence explanation. "
           "Panel members ලෙස result immediately understand කිරීමට add කළෙමු."), sp(0.2)]

    # Page 4 Model Comparison
    story += [Paragraph("PAGE 4 — 📈 Model Comparison", EH),
        bi("Compares 1D-CNN-BiLSTM vs 1D-ResNet vs Standard-CNN vs classical baselines "
           "(Random Forest, SVM, KNN). All evaluated on the same zero-leakage N=150 test set.",
           "1D-CNN-BiLSTM vs 1D-ResNet vs Standard-CNN vs classical baselines "
           "(Random Forest, SVM, KNN) compare. All zero-leakage N=150 test set."),
        sp(0.15),
        qa("Why is CNN-BiLSTM better than pure CNN?",
           "CNN layers extract local spectral features (absorption line shapes at specific wavelengths). "
           "BiLSTM adds bidirectional sequential context — it understands how features across the "
           "full 4,000-channel spectrum relate to each other. Pure CNN misses this long-range context.",
           "CNN layers local spectral features (specific wavelengths absorption line shapes) extract. "
           "BiLSTM bidirectional sequential context add — full 4,000-channel spectrum features relate "
           "ආකාරය understand. Pure CNN long-range context miss කරයි."), sp(0.2)]

    # Page 5 XAI Validation
    story += [Paragraph("PAGE 5 — 🧪 Scientific XAI Validation", EH),
        bi("Shows AOPC Faithfulness test results and Jaccard Stability test results. "
           "Also shows the NIST line matching results and Monte Carlo significance test.",
           "AOPC Faithfulness test results, Jaccard Stability test results, "
           "NIST line matching, Monte Carlo significance test show කරයි."),
        sp(0.15),
        qa("What is AOPC and what does 0.7507 mean?",
           "AOPC = Area Over the Perturbation Curve. We masked the top-K attribution channels "
           "and measured confidence drop. Score 0.7507 for top-K masking vs 0.1192 for random "
           "masking. Ratio 6.30× — attribution-guided masking causes 6× more confidence drop "
           "than random. This confirms attributions point to genuinely important regions.",
           "AOPC = Area Over Perturbation Curve. Top-K attribution channels mask කර "
           "confidence drop measure. Score 0.7507 (top-K) vs 0.1192 (random). "
           "Ratio 6.30× — attribution masking random ලෙස 6× confidence drop. "
           "Attributions genuinely important regions indicate ලෙස confirm."),
        sp(0.15),
        qa("What is Jaccard Stability and what does 0.8069 mean?",
           "We added Gaussian noise (sigma=0.01, 0.02, 0.04) to test spectra and re-computed "
           "SHAP attributions. Jaccard similarity measures overlap of top-K peaks between clean "
           "and noisy attributions. 0.8069 at sigma=0.01 means 80.7% of top attribution peaks "
           "remain the same after realistic noise — explanations are stable.",
           "Gaussian noise (sigma=0.01, 0.02, 0.04) test spectra ලෙස add කර SHAP re-compute. "
           "Jaccard similarity — clean vs noisy attributions top-K peaks overlap. "
           "0.8069 at sigma=0.01 — realistic noise ලෙස top peaks 80.7% same — explanations stable."),
        sp(0.15),
        qa("What does Monte Carlo p=0.538 mean?",
           "We generated 1,000 random peak sets and measured how often they matched NIST lines "
           "as well as our SHAP peaks (10.53% match rate). p=0.538 means 53.8% of random sets "
           "matched equally well — NOT statistically significant. Our line matching is exploratory "
           "candidate evidence, not a confirmed discovery.",
           "Random peak sets 1,000 generate කර SHAP peaks (10.53% match) ලෙස NIST lines "
           "match ආකාරය measure. p=0.538 — random sets 53.8% equally match — "
           "statistically significant නොවේ. Line matching exploratory candidate evidence — "
           "confirmed discovery නොවේ."), sp(0.2),
        PageBreak(),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 4 — SPECTRAL CLASSES DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    story += [banner("CHAPTER 4 — SPECTRAL CLASSES — Deep Dive Q&A",
                     "Spectral Classes — Q&A", CB), sp(0.3)]

    classes_qa = [
        ("K", "Orange/Red Giant", "3,500–5,000 K",
         "Aging, evolved, post-main-sequence giant. Core hydrogen exhausted. Rich in Mg I, Si I, "
         "Fe I and strong CO molecular bands (carbon monoxide) in NIR.",
         "Asymptotic Giant Branch → Planetary Nebula → C-O White Dwarf",
         "Core hydrogen exhausted → giant ලෙස expand → Mg I, Si I, Fe I neutral metals + CO molecular bands NIR dominant",
         "Asymptotic Giant Branch → Planetary Nebula → Carbon-Oxygen White Dwarf",
         "Aging Post-Main-Sequence Red Giant — core hydrogen exhausted, dramatically expanded, CO molecular bands.",
         "Core hydrogen exhausted Red Giant — dramatically expand, CO molecular bands dominant."),

        ("G", "Solar-Type Star", "5,000–6,000 K",
         "Solar twin. Stable Main Sequence hydrogen-burning star. Fe I, Ca II, Si I, O I absorption. "
         "Galactic thin disk population. Like our own Sun (T_sun ≈ 5,778 K).",
         "Red Giant expansion (~5 billion years) → C-O White Dwarf",
         "Stable Main Sequence, hydrogen burning. Fe I, Ca II, Si I, O I absorption. Galactic thin disk.",
         "Red Giant expansion (billion years ~5) → Carbon-Oxygen White Dwarf",
         "Solar Twin — active stable yellow dwarf identical to our Sun.",
         "Solar Twin — Sun ලෙස identical stable hydrogen-burning star."),

        ("F", "Intermediate Star", "6,000–7,500 K",
         "Warmer than the Sun. Main Sequence / early post-main-sequence. "
         "Hydrogen Brackett lines + ionized metals (Fe II, Ca II) prominent. "
         "Young to intermediate disk population.",
         "Red Giant → White Dwarf",
         "Sun ලෙස warmer. Hydrogen Brackett + ionized Fe II, Ca II prominent. Young/intermediate disk.",
         "Red Giant → White Dwarf",
         "Warm Intermediate White Star — warmer and brighter than the Sun.",
         "Intermediate White Star — Sun ලෙස warmer/brighter."),

        ("A", "Hot Early-Type Star", "7,500–10,000 K",
         "Hot white star. Short lifetime (~1 billion years). Strong Hydrogen Brackett series "
         "(NIR H-band) dominant. Young star-forming regions.",
         "Massive Red Giant → High-mass White Dwarf",
         "Hot white. Lifetime ~billion years 1. Strong Hydrogen Brackett series NIR dominant. Star-forming regions.",
         "Massive Red Giant → High-mass White Dwarf",
         "Luminous Hot Early-Type — brief lifetime, intense Hydrogen Brackett absorption.",
         "Hot Early-Type — brief lifetime, intense Hydrogen Brackett absorption."),

        ("B", "Luminous Blue Giant", "10,000–30,000 K",
         "Extremely hot, massive, young. He I (neutral helium) lines + Brackett series. "
         "Very rare in APOGEE — found in young spiral arms and OB associations.",
         "Core collapse Supernova or Massive White Dwarf",
         "Extremely hot, massive, young. He I + Brackett. APOGEE ලෙස very rare — spiral arms, OB associations.",
         "Supernova (core collapse) හෝ Massive White Dwarf",
         "Extremely Hot Massive Blue Giant — rapid nuclear burning, destined for Supernova.",
         "Hot Massive Blue Giant — rapid nuclear consumption, Supernova ඉලක්ක."),
    ]

    for cls, ctype, teff, en_desc, en_fate, sin_desc, sin_fate, en_take, sin_take in classes_qa:
        data = [[
            Paragraph(f"<font color='white'><b>Class {cls}</b><br/>{ctype}</font>",
                      S(f'cl{cls}', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=CW)),
            Paragraph(f"<b>Teff:</b> {teff}<br/><b>Description:</b> {en_desc}<br/>"
                      f"<b>Primary Finding:</b> {en_take}<br/><b>Stellar Fate:</b> {en_fate}", EB),
            Paragraph(f"<b>Teff:</b> {teff}<br/>{sin_desc}<br/>"
                      f"<b>ප්‍රධාන නිගමනය:</b> {sin_take}<br/><b>Fate:</b> {sin_fate}", SB),
        ]]
        t = Table(data, colWidths=[72, 222, 221])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(0,-1),CN), ('BACKGROUND',(1,0),(1,-1),colors.HexColor('#EEF4FF')),
            ('BACKGROUND',(2,0),(2,-1),CGB),
            ('BOX',(0,0),(-1,-1),0.5,CBD), ('INNERGRID',(0,0),(-1,-1),0.5,CBD),
            ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),5), ('VALIGN',(0,0),(-1,-1),'TOP'),
        ]))
        story += [t, sp(0.1)]

    story += [sp(0.2), PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 5 — ALL EXPECTED PANEL QUESTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    story += [banner("CHAPTER 5 — ALL EXPECTED PANEL QUESTIONS & ANSWERS",
                     "Panel ප්‍රශ්න සම්පූර්ණ Q&A", CA), sp(0.3),
        info("These are the most likely questions your PP1 panel will ask. Practise each answer.",
             "PP1 panel ඇසිය හැකි සෑම ප්‍රශ්නයක්ම මෙහිය. සෑම පිළිතුරක්ම practice කරන්න."),
        sp(0.3)]

    panel_qs = [
        # --- Research & Motivation ---
        ("What is your research question?",
         "Can Explainable AI (Gradient SHAP) reliably identify physically meaningful atomic "
         "absorption features in near-infrared stellar spectra classified by a 1D-CNN-BiLSTM model?",
         "Explainable AI (Gradient SHAP) 1D-CNN-BiLSTM model classify කළ near-infrared stellar spectra "
         "හි physically meaningful atomic absorption features reliably identify කළ හැකිද?"),

        ("Why is stellar spectra classification important?",
         "Stellar classification enables us to determine a star's temperature, age, chemical "
         "composition, and evolutionary state. With millions of spectra from surveys like APOGEE/SDSS, "
         "automated AI classification is essential. XAI adds scientific trustworthiness by showing "
         "which physical features drove each classification.",
         "Stellar classification ලෙස star temperature, age, chemical composition, evolutionary state "
         "determine කළ හැකිය. APOGEE/SDSS surveys ලෙස millions of spectra ඇති — automated AI "
         "classification essential. XAI physical features show කර scientific trustworthiness add."),

        ("Why APOGEE data specifically?",
         "APOGEE (Apache Point Observatory Galactic Evolution Experiment) provides high-resolution "
         "near-infrared (H-band: 15,100–17,000 Å) spectra of ~650,000 stars. NIR penetrates dust, "
         "enabling observation of the Galactic bulge and disk. Our 1D-CNN processes the 4,000-channel "
         "log-wavelength grid directly.",
         "APOGEE near-infrared (H-band: 15,100–17,000 Å) high-resolution spectra stars ~650,000 "
         "provide. NIR dust penetrate — Galactic bulge/disk observation. "
         "1D-CNN 4,000-channel log-wavelength grid directly process."),

        # --- Model ---
        ("Why 1D-CNN-BiLSTM and not Transformer?",
         "Transformers require large datasets and are computationally expensive. Our dataset has "
         "only 1,000 corrected samples. 1D-CNN-BiLSTM is well-suited for sequential spectral data — "
         "CNN extracts local absorption patterns; BiLSTM captures long-range dependencies. "
         "It achieves 77.33% accuracy with ~350K parameters — efficient and effective.",
         "Transformers large datasets + computationally expensive. Dataset corrected samples 1,000 only. "
         "1D-CNN-BiLSTM sequential spectral data ලෙස suited — CNN local absorption patterns; "
         "BiLSTM long-range dependencies. 77.33% accuracy, ~350K parameters — efficient."),

        ("Did you retrain after finding the data leakage bug?",
         "Yes. After identifying the nested-indexing data leakage bug in 01_data_preprocess.py, "
         "we regenerated the corrected zero-leakage splits (700/150/150) and retrained "
         "train_corrected_split.py from scratch. The corrected model checkpoint is saved at "
         "models/corrected_split/1d_cnn_bilstm.pth.",
         "ඔව්. 01_data_preprocess.py nested-indexing data leakage bug identify කළ පසු, "
         "corrected zero-leakage splits (700/150/150) regenerate කර "
         "train_corrected_split.py scratch ලෙස retrain කළෙමු. "
         "Corrected checkpoint models/corrected_split/1d_cnn_bilstm.pth."),

        ("What is the class distribution in your dataset?",
         "Class K: ~87% (dominant — APOGEE targets red giants), G: ~8%, F: ~3%, A: ~1%, B: <1%. "
         "This severe imbalance is why we report balanced accuracy (74.15%) and Macro F1 (0.6406) "
         "alongside raw accuracy (77.33%).",
         "Class K: ~87% (dominant — APOGEE red giants target), G: ~8%, F: ~3%, A: ~1%, B: <1%. "
         "Severe imbalance — balanced accuracy (74.15%) සහ Macro F1 (0.6406) "
         "raw accuracy (77.33%) ලෙස report."),

        # --- XAI ---
        ("What is Gradient SHAP?",
         "Gradient SHAP is an XAI method that computes expected gradients over a set of background "
         "reference samples to assign attribution values to each input feature. In our case, each of "
         "the 4,000 wavelength channels gets an attribution score showing how much it influenced "
         "the model's prediction for that spectrum.",
         "Gradient SHAP background reference samples set ලෙස expected gradients compute කර "
         "input features attribution values assign කරන XAI method. "
         "Wavelength channels 4,000 ලෙස attribution score — prediction influence ප්‍රමාණය."),

        ("Why Gradient SHAP and not Grad-CAM or LIME?",
         "Grad-CAM works on 2D CNNs (images) — less suitable for 1D spectra. "
         "LIME is model-agnostic but slow and approximation-based. "
         "Gradient SHAP is model-specific to neural networks, exact (not approximate), "
         "fast for 4,000-channel inputs, and directly produces per-wavelength attributions "
         "comparable to known spectral lines.",
         "Grad-CAM 2D CNNs (images) — 1D spectra ලෙස less suitable. "
         "LIME model-agnostic නමුත් slow + approximation. "
         "Gradient SHAP neural networks model-specific, exact, fast 4,000-channel, "
         "known spectral lines compare per-wavelength attributions produce."),

        ("What are the 6 NIST matched lines?",
         "O I (15,871 Å), H Br14 (15,263 Å), N I (16,764 Å), Si I (15,960 Å), "
         "Mg I (15,745 Å), O I (15,569 Å). These are 6 of 57 SHAP peaks (10.53%). "
         "Monte Carlo test: p=0.538 — not statistically significant. Exploratory only.",
         "O I (15,871 Å), H Br14 (15,263 Å), N I (16,764 Å), Si I (15,960 Å), "
         "Mg I (15,745 Å), O I (15,569 Å). SHAP peaks 57 ලෙස 6 (10.53%). "
         "Monte Carlo p=0.538 — statistically significant නොවේ. Exploratory only."),

        # --- Limitations ---
        ("What are the main limitations of your system?",
         "1) Class labels are TEFF-based proxy classes, not verified Harvard types. "
         "2) Dataset has severe class imbalance (K: ~87%). "
         "3) NIST line matching is exploratory (10.53%, p=0.538). "
         "4) Stability tests used synthetic Gaussian noise, not real observing conditions. "
         "5) Raw FITS files not included in submission. "
         "6) System only accepts APOGEE H-band spectra — not generalised to other instruments.",
         "1) Class labels TEFF-based proxy — verified Harvard types නොවේ. "
         "2) Dataset severe class imbalance (K: ~87%). "
         "3) NIST line matching exploratory (10.53%, p=0.538). "
         "4) Stability tests synthetic noise — real conditions නොවේ. "
         "5) Raw FITS files submission ලෙස නැත. "
         "6) APOGEE H-band only — other instruments generalised නොවේ."),

        ("What would you improve in future work?",
         "1) Collect more Class A, B, F spectra to reduce imbalance. "
         "2) Implement per-query uncertainty quantification (MC Dropout / Deep Ensembles). "
         "3) Add Grad-CAM++ comparison to cross-validate SHAP attribution peaks. "
         "4) Extend to SDSS optical spectra (4,000–7,000 Å) for generalisability. "
         "5) Replace proxy TEFF class labels with expert-verified Harvard spectral types.",
         "1) Class A, B, F spectra collect — imbalance reduce. "
         "2) Per-query uncertainty quantification (MC Dropout / Deep Ensembles). "
         "3) Grad-CAM++ comparison — SHAP peaks cross-validate. "
         "4) SDSS optical spectra (4,000–7,000 Å) — generalisability. "
         "5) Proxy TEFF labels → expert-verified Harvard spectral types."),

        # --- Upload page specific ---
        ("Does your system work on stars it has never seen before?",
         "Yes. The model was trained to learn spectral absorption patterns (physics-based features), "
         "not to memorise individual stars. Any new APOGEE H-band spectrum (15,100–17,000 Å) "
         "preprocessed to 4,000 channels will be classified. Confidence score indicates reliability.",
         "ඔව්. Model individual stars memorise නොව spectral absorption patterns (physics-based) "
         "learn කිරීමට train. New APOGEE H-band spectrum (15,100–17,000 Å) "
         "channels 4,000 ලෙස preprocess — classify. Confidence score reliability indicate."),

        ("What happens when a user uploads an image FITS (like Hubble photos)?",
         "The system detects a numpy structured record array (2D image/binary table metadata) "
         "and shows a clear user-friendly error: 'Unsupported File Format — appears to be an "
         "HST/JWST imaging FITS, not a 1D stellar spectrum.' It lists accepted formats and "
         "provides sample file hints. The app does not crash.",
         "System numpy structured record array (2D image/binary table) detect — "
         "clear error: 'Unsupported File Format — HST/JWST imaging FITS, 1D spectrum නොවේ.' "
         "Accepted formats list, sample file hints. App crash නොවේ."),

        ("How many automated tests does your system have?",
         "70 automated tests across test_pipeline.py and test_extended_audit.py, covering: "
         "dataset integrity, zero-leakage split validation, confusion matrix consistency, "
         "SHAP tensor shape, NIST line CSV validation, preprocessing contract tests (CSV/NPZ/FITS/NPY), "
         "format boundary tests, and PDF report generation. All 70 pass.",
         "Automated tests 70 — test_pipeline.py + test_extended_audit.py: "
         "dataset integrity, zero-leakage split, confusion matrix, SHAP tensor shape, "
         "NIST line CSV, preprocessing contracts (CSV/NPZ/FITS/NPY), format boundary, "
         "PDF report generation. 70 සියල්ල pass."),

        ("What is F1 score and why is Macro F1 used?",
         "F1 = 2 × (Precision × Recall) / (Precision + Recall). It balances how often the model "
         "is right when it predicts a class (precision) with how often it finds all true members "
         "of that class (recall). Macro F1 averages F1 equally across all 5 classes — "
         "it penalises poor performance on minority classes (A, B, F) equally with majority (K). "
         "Raw accuracy would hide poor minority class performance.",
         "F1 = 2 × (Precision × Recall) / (Precision + Recall). "
         "Precision (predict කළ class නිවැරදි proportion) + Recall (true members find proportion) balance. "
         "Macro F1 classes 5 equally average — minority classes (A, B, F) poor performance penalise. "
         "Raw accuracy minority class poor performance hide කරයි."),
    ]

    for q, a_en, a_sin in panel_qs:
        story += [qa(q, a_en, a_sin), sp(0.2)]

    story += [PageBreak()]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAPTER 6 — KEY NUMBERS TO MEMORISE
    # ═══════════════════════════════════════════════════════════════════════════
    story += [banner("CHAPTER 6 — KEY NUMBERS TO MEMORISE",
                     "Key Numbers — වශයෙන් ඉගෙන ගන්න", CG), sp(0.3)]

    numbers = [
        ["Metric / Value", "Number", "What it means (EN)", "සිංහල"],
        ["Test Accuracy",            "77.33%",  "Correct predictions on 150-sample test set", "Test set 150 ලෙස correct predictions"],
        ["Balanced Accuracy",        "74.15%",  "Average accuracy per class (handles imbalance)", "Per-class average accuracy"],
        ["Macro F1",                 "0.6406",  "Equal-weight F1 across all 5 classes", "Classes 5 ම equal-weight F1"],
        ["Train / Val / Test",       "700/150/150","Zero-leakage corrected split",     "Zero-leakage split"],
        ["AOPC Top-K",               "0.7507",  "Confidence drop from masking top attributions", "Top attribution mask → confidence drop"],
        ["AOPC Random",              "0.1192",  "Confidence drop from random masking", "Random mask → confidence drop"],
        ["AOPC Ratio",               "6.30×",   "Top-K masking causes 6.3× more drop than random", "Top-K masking random ලෙස 6.3× more drop"],
        ["Jaccard σ=0.01",           "0.8069",  "80.7% of top SHAP peaks stable at low noise", "Low noise ලෙස SHAP peaks 80.7% stable"],
        ["Jaccard σ=0.02",           "0.7768",  "77.7% stable at medium noise",        "Medium noise ලෙස 77.7% stable"],
        ["Jaccard σ=0.04",           "0.6853",  "68.5% stable at high noise",          "High noise ලෙස 68.5% stable"],
        ["NIST Matches",             "6 / 57",  "10.53% of SHAP peaks matched NIST lines","SHAP peaks 10.53% NIST match"],
        ["Monte Carlo p-value",      "p = 0.538","Not statistically significant",       "Statistically significant නොවේ"],
        ["Model Parameters",         "~350K",   "1D-CNN-BiLSTM trainable parameters", "Trainable parameters"],
        ["Wavelength Range",         "15,100–17,000 Å","APOGEE H-band near-infrared", "APOGEE H-band near-infrared"],
        ["Grid Points",              "4,000",   "Log-spaced wavelength channels per spectrum", "Spectrum per log-spaced channels"],
        ["Historical Accuracy",      "91.33%",  "Old buggy leaky result — DO NOT CITE THIS", "Leaky bug result — CITE කරන්න එපා"],
        ["Total Automated Tests",    "70",      "All passing in test suite",           "Test suite pass 70"],
        ["Matched NIST Lines",       "O I, H Br14, N I, Si I, Mg I, O I", "6 verified NIST matches","NIST matches 6"],
    ]
    t = Table([[Paragraph(f"<b>{c}</b>" if i==0 else c,
                          EBB if i==0 else (SB if j==3 else (
                          S('red', fontName='Helvetica-Bold', fontSize=9, textColor=CR)
                          if ("91.33" in c or "CITE" in c) else EB)))
                for j,c in enumerate(row)] for i,row in enumerate(numbers)],
              colWidths=[120, 60, 200, 135])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),CG), ('TEXTCOLOR',(0,0),(-1,0),CW),
        ('INNERGRID',(0,0),(-1,-1),0.5,CBd()), ('BOX',(0,0),(-1,-1),0.5,CBd()),
        ('TOPPADDING',(0,0),(-1,-1),3), ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('LEFTPADDING',(0,0),(-1,-1),4),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[CLG,CW]),
        ('BACKGROUND',(0,17),(-1,17),CRB),
    ]))
    story += [t, sp(0.4), hr(),
        Paragraph("J26-IT-449  |  P.K. Edirisooriya (IT23330696)  |  SLIIT IT4010 PP1  |  Generated: " +
                  time.strftime("%Y-%m-%d"), FT)]

    # ── Build ─────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=1.4*cm, leftMargin=1.4*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm,
        title="Full PP1 Viva Panel Guide", author="P.K. Edirisooriya IT23330696")
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


if __name__ == '__main__':
    print("Generating Full PP1 Viva Panel Guide...")
    pdf_bytes = build()
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(pdf_bytes)
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Size  -> {len(pdf_bytes)/1024:.1f} KB")
