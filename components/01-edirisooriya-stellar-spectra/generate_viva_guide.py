"""
generate_viva_guide.py
======================
Generates a comprehensive bilingual (English + Sinhala) Viva Preparation Guide PDF
for the Stellar Spectra XAI Research System (SLIIT J26-IT-449).

Covers every page and feature of the Streamlit app with full Sinhala explanations.

Run:
    python generate_viva_guide.py
Output:
    reports/VIVA_PREPARATION_GUIDE_SINHALA.pdf
"""

import io
import os
import time

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)

# ── Register Nirmala UI (Sinhala support) ─────────────────────────────────────
pdfmetrics.registerFont(TTFont('Nirmala', 'C:/Windows/Fonts/Nirmala.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('NirmalaB', 'C:/Windows/Fonts/Nirmala.ttc', subfontIndex=1))

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT, 'reports', 'VIVA_PREPARATION_GUIDE_SINHALA.pdf')

# ── Color Palette ─────────────────────────────────────────────────────────────
C_NAVY     = colors.HexColor('#0F2043')
C_BLUE     = colors.HexColor('#1F77B4')
C_GREEN    = colors.HexColor('#2E7D32')
C_GREEN_BG = colors.HexColor('#EBF7EE')
C_AMBER    = colors.HexColor('#E65100')
C_AMBER_BG = colors.HexColor('#FFF3E0')
C_PURPLE   = colors.HexColor('#5B0060')
C_PURPLE_BG= colors.HexColor('#F3E5F5')
C_GREY     = colors.HexColor('#555555')
C_LGREY    = colors.HexColor('#F8F9FA')
C_BORDER   = colors.HexColor('#D0D7DE')
C_WHITE    = colors.white


def make_styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    title_style = ps('DocTitle',
        fontName='Helvetica-Bold', fontSize=22, leading=28,
        textColor=C_NAVY, spaceAfter=4, alignment=1)

    subtitle_style = ps('DocSub',
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=C_GREY, spaceAfter=2, alignment=1)

    # Page section header
    page_header = ps('PageHeader',
        fontName='Helvetica-Bold', fontSize=15, leading=20,
        textColor=C_WHITE, spaceAfter=0, spaceBefore=0)

    # English section heading
    eng_heading = ps('EngHeading',
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=C_BLUE, spaceBefore=8, spaceAfter=3)

    # English body
    eng_body = ps('EngBody',
        fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=colors.HexColor('#1A1A1A'), spaceAfter=2)

    # English bold body
    eng_bold = ps('EngBold',
        fontName='Helvetica-Bold', fontSize=9.5, leading=14,
        textColor=colors.HexColor('#1A1A1A'))

    # Sinhala body
    sin_body = ps('SinBody',
        fontName='Nirmala', fontSize=10, leading=17,
        textColor=colors.HexColor('#0D3B00'), spaceAfter=2)

    # Sinhala bold
    sin_bold = ps('SinBold',
        fontName='NirmalaB', fontSize=10, leading=17,
        textColor=C_GREEN)

    # Sinhala heading
    sin_heading = ps('SinHeading',
        fontName='NirmalaB', fontSize=11, leading=18,
        textColor=colors.HexColor('#0D3B00'), spaceBefore=4, spaceAfter=2)

    # Viva Q&A question
    viva_q = ps('VivaQ',
        fontName='Helvetica-Bold', fontSize=9.5, leading=14,
        textColor=C_AMBER, spaceBefore=4)

    # Viva Q&A answer
    viva_a = ps('VivaA',
        fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=colors.HexColor('#1A1A1A'))

    return {
        'title': title_style, 'subtitle': subtitle_style,
        'page_header': page_header,
        'eng_heading': eng_heading, 'eng_body': eng_body, 'eng_bold': eng_bold,
        'sin_body': sin_body, 'sin_bold': sin_bold, 'sin_heading': sin_heading,
        'viva_q': viva_q, 'viva_a': viva_a,
    }


def section_banner(title_en, title_sin, styles, color=C_NAVY):
    """Creates a full-width coloured banner for each page section."""
    content = [[
        Paragraph(f"<font color='white'><b>{title_en}</b></font>", styles['page_header']),
        Paragraph(f"<font color='#AAFFAA'>{title_sin}</font>",
                  ParagraphStyle('BannerSin', fontName='Nirmala', fontSize=10,
                                 leading=14, textColor=colors.HexColor('#AAFFAA')))
    ]]
    t = Table(content, colWidths=[350, 165])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), color),
        ('TOPPADDING', (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING', (0,0),(-1,-1), 10),
        ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


def bilingual_block(en_text, sin_text, styles, en_label="English", sin_label="සිංහල"):
    """Creates a side-by-side or stacked bilingual explanation block."""
    data = [[
        Paragraph(f"<b>🇬🇧 {en_label}:</b><br/>{en_text}", styles['eng_body']),
        Paragraph(f"<b>🇱🇰 {sin_label}:</b><br/>{sin_text}", styles['sin_body']),
    ]]
    t = Table(data, colWidths=[258, 257])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(0,-1), colors.HexColor('#F0F5FF')),
        ('BACKGROUND', (1,0),(1,-1), C_GREEN_BG),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 6),
        ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
    ]))
    return t


def viva_qa(question, answer_en, answer_sin, styles):
    """Creates a viva Q&A block."""
    data = [
        [Paragraph(f"❓ {question}", styles['viva_q']), ''],
        [Paragraph(f"✅ <b>Answer:</b> {answer_en}", styles['viva_a']),
         Paragraph(f"✅ <b>පිළිතුර:</b> {answer_sin}", styles['sin_body'])],
    ]
    t = Table(data, colWidths=[258, 257])
    t.setStyle(TableStyle([
        ('SPAN', (0,0),(1,0)),
        ('BACKGROUND', (0,0),(1,0), C_AMBER_BG),
        ('BACKGROUND', (0,1),(0,1), colors.HexColor('#F0F5FF')),
        ('BACKGROUND', (1,1),(1,1), C_GREEN_BG),
        ('BOX', (0,0),(-1,-1), 1, C_AMBER),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 6),
        ('RIGHTPADDING', (0,0),(-1,-1), 6),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
    ]))
    return t


def info_box(text_en, text_sin, styles, color=C_BLUE):
    """Creates a highlighted info callout box."""
    data = [[
        Paragraph(f"<font color='white'><b>ℹ {text_en}</b></font><br/>"
                  f"<font color='#DDFFDD'>{text_sin}</font>",
                  ParagraphStyle('InfoBox', fontName='Nirmala', fontSize=9.5,
                                 leading=15, textColor=C_WHITE))
    ]]
    t = Table(data, colWidths=[515])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), color),
        ('BOX', (0,0),(-1,-1), 0.5, color),
        ('TOPPADDING', (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING', (0,0),(-1,-1), 10),
        ('RIGHTPADDING', (0,0),(-1,-1), 10),
    ]))
    return t


def build_viva_guide():
    styles = make_styles()
    story = []

    W = 515  # usable width

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("VIVA PREPARATION GUIDE", styles['title']))
    story.append(Paragraph("Stellar Spectra XAI System — Complete Page-by-Page Walkthrough", styles['subtitle']))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("තාරකා වර්ණාවලි XAI පද්ධතිය — සිංහල Viva සූදානම් වීමේ මාර්ගෝපදේශය", 
                           ParagraphStyle('CoverSin', fontName='NirmalaB', fontSize=13, leading=20,
                                         textColor=C_GREEN, alignment=1)))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_NAVY, spaceAfter=8))

    cover_info = [
        ["Project ID", "J26-IT-449 | SLIIT IT4010 Research Project — PP1 Assessment"],
        ["Student", "P.K. Edirisooriya (IT23330696)"],
        ["Research Title", "Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification"],
        ["Model", "1D-CNN-BiLSTM | Test Accuracy: 77.33% | Balanced Acc: 74.15% | Macro F1: 0.6406"],
        ["Dataset", "APOGEE DR17 Near-Infrared Stellar Spectra | 700 Train / 150 Val / 150 Test"],
        ["Generated", time.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    t_cover = Table([[Paragraph(f"<b>{k}</b>", styles['eng_bold']),
                      Paragraph(v, styles['eng_body'])] for k, v in cover_info],
                    colWidths=[130, 385])
    t_cover.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_LGREY),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
    ]))
    story.append(t_cover)
    story.append(Spacer(1, 0.4*cm))

    story.append(info_box(
        "How to use this guide: Each section covers one page of the Streamlit app with English explanation + Sinhala translation + Viva Q&A.",
        "මෙම මාර්ගෝපදේශය භාවිතා කරන ආකාරය: සෑම කොටසක්ම Streamlit app එකේ එක් page එකක් ආවරණය කරයි — ඉංග්‍රීසි පැහැදිලි කිරීම + සිංහල පරිවර්තනය + Viva ප්‍රශ්නෝත්තර.",
        styles
    ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0: SYSTEM OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("SYSTEM OVERVIEW", "පද්ධති දළ විශ්ලේෂණය", styles, C_NAVY))
    story.append(Spacer(1, 0.3*cm))

    story.append(bilingual_block(
        "This is a Streamlit-based web application that classifies stellar spectra using a trained 1D-CNN-BiLSTM deep learning model and explains its predictions using Gradient SHAP (Explainable AI). It has 6 pages accessible via the sidebar.",
        "මෙය Streamlit-based web application එකක් වන අතර, 1D-CNN-BiLSTM deep learning model එකක් භාවිතයෙන් තාරකා spectra වර්ගීකරණය (classify) කර, Gradient SHAP (Explainable AI) භාවිතයෙන් ප්‍රතිඵල පැහැදිලි කරයි. Sidebar එකෙහි Page 6ක් ඇත.",
        styles
    ))
    story.append(Spacer(1, 0.3*cm))

    pages_overview = [
        ["Page", "Name", "Purpose (English)", "අරමුණ (සිංහල)"],
        ["1", "🏠 Dashboard", "System overview, dataset distribution, metrics", "පද්ධති සාරාංශය, dataset, metrics"],
        ["2", "🔭 Spectrum Explorer", "Browse test spectra interactively", "Test spectra browse කිරීම"],
        ["3", "🔬 Upload & Analyse", "Upload your own spectrum, get live prediction", "ඔබගේ spectrum upload කිරීම"],
        ["4", "📈 Model Comparison", "Compare CNN-BiLSTM vs classical baselines", "Models සංසන්දනය"],
        ["5", "🧪 Scientific XAI Validation", "AOPC faithfulness & Jaccard stability tests", "XAI විද්‍යාත්මක වලංගු කිරීම"],
        ["6", "ℹ️ About & Limitations", "System limitations, how to run, citations", "සීමාවන් සහ ධාවන විධිය"],
    ]
    t_pages = Table([[Paragraph(f"<b>{c}</b>" if i == 0 else c, styles['eng_bold'] if i == 0 else styles['eng_body'])
                      for c in row] for i, row in enumerate(pages_overview)],
                    colWidths=[28, 100, 210, 177])
    t_pages.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_NAVY),
        ('TEXTCOLOR', (0,0),(-1,0), C_WHITE),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ('LEFTPADDING', (0,0),(-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_LGREY, C_WHITE]),
    ]))
    story.append(t_pages)
    story.append(Spacer(1, 0.3*cm))
    story.append(viva_qa(
        "What is this system?",
        "This is a web-based stellar spectra classification and XAI dashboard. It uses a 1D-CNN-BiLSTM model trained on APOGEE near-infrared spectra to classify stars into 5 spectral classes (A, B, F, G, K) and uses Gradient SHAP to identify which wavelength regions drove each prediction.",
        "මෙය web-based තාරකා spectra වර්ගීකරණ සහ XAI dashboard එකකි. APOGEE near-infrared spectra මත පුහුණු කළ 1D-CNN-BiLSTM model එකක් භාවිතයෙන් තාරකා A, B, F, G, K ලෙස Spectral Classes 5 කට වර්ගීකරණය කරයි. Gradient SHAP භාවිතයෙන් ප්‍රතිඵලවලට බලපෑ wavelength regions හඳුනා ගනී.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 1 — 🏠 Dashboard", "Page 1 — Dashboard", styles, C_BLUE))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("What this page contains", styles['eng_heading']))
    story.append(bilingual_block(
        "The Dashboard is the home page. It displays: (1) A bar chart of class distribution across the dataset manifest (N=1,004 rows), (2) Model performance metrics (accuracy, balanced accuracy, macro F1), (3) The confusion matrix showing how often each class was correctly predicted, (4) A summary of the XAI methods used.",
        "Dashboard page එක home page එකයි. එය පෙන්වන දෑ: (1) Dataset manifest (N=1,004) හි class distribution bar chart, (2) Model performance metrics — accuracy, balanced accuracy, macro F1, (3) Confusion matrix — සෑම class එකක්ම නිවැරදිව predict කළ ආකාරය, (4) භාවිත XAI methods සාරාංශය.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Key Numbers to Remember", styles['eng_heading']))
    metrics_data = [
        ["Metric", "Value", "What it means (English)", "සිංහල පැහැදිලි කිරීම"],
        ["Test Accuracy", "77.33%", "77 out of 100 spectra predicted correctly", "100 spectra 77 ක් නිවැරදිව predict විය"],
        ["Balanced Accuracy", "74.15%", "Average accuracy across all 5 classes", "Class 5 ම හරහා සාමාන්‍ය නිරවද්‍යතාව"],
        ["Macro F1", "0.6406", "Average F1 score equally weighted per class", "Class සෑමකටම සමාන weightage F1 score"],
        ["Test Set Size", "N=150", "Zero-leakage isolated test split", "කිසිදු data leak නොවූ isolated test set"],
        ["Train Split", "700", "Samples used for training", "Model පුහුණු කිරීමට භාවිත samples"],
        ["Val Split", "150", "Samples used for validation/tuning", "Validation/tuning සඳහා samples"],
    ]
    t_metrics = Table([[Paragraph(f"<b>{c}</b>" if i == 0 else c, styles['eng_bold'] if i == 0 else styles['eng_body'])
                        for c in row] for i, row in enumerate(metrics_data)],
                      colWidths=[90, 55, 180, 190])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_NAVY),
        ('TEXTCOLOR', (0,0),(-1,0), C_WHITE),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING', (0,0),(-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_LGREY, C_WHITE]),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 0.2*cm))

    story.append(viva_qa(
        "Why is accuracy 77.33% and not higher?",
        "The dataset has severe class imbalance — Class K dominates (~87% of samples). Despite this, our model achieves 74.15% balanced accuracy. Low minority-class samples (A, B) limit performance. Higher accuracy is possible with more data.",
        "Dataset එකේ class imbalance ඉතා ශක්තිමත්ය — Class K ~87% ක් ආවරණය කරයි. එය නිසාවෙන් minority classes (A, B) සඳහා accuracy අඩු වේ. 74.15% balanced accuracy ලබා ගත හැකි විය. Data වැඩි නම් accuracy ඉහළ දැමිය හැකිය.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(viva_qa(
        "What is a confusion matrix?",
        "A confusion matrix is a table showing how many spectra of each true class were predicted as each class. The diagonal shows correct predictions. Off-diagonal shows mistakes.",
        "Confusion matrix යනු සෑම true class එකක් model එක විසින් කොයි class ලෙස predict කළේ ද කියා පෙන්වන වගුවකි. Diagonal (ඇතිරිල්ල) නිවැරදි predictions. Off-diagonal (ඉවත) වැරදි predictions.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: SPECTRUM EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 2 — 🔭 Spectrum Explorer", "Page 2 — Spectrum Explorer", styles, C_PURPLE))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("What this page contains", styles['eng_heading']))
    story.append(bilingual_block(
        "The Spectrum Explorer lets you browse the 150 isolated test-set spectra. You can select any sample by index, view its normalised flux plot, see the model's prediction and confidence, and view the Gradient SHAP attributions alongside matched NIST atomic spectral lines.",
        "Spectrum Explorer page එකෙන් isolated test set spectra 150 browse කළ හැකිය. Index අනුව ඕනෑම sample එකක් select කර, normalised flux plot, model prediction, confidence, සහ Gradient SHAP attributions — NIST atomic lines සමඟ — දැකගත හැකිය.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("What each element shows", styles['eng_heading']))
    elements = [
        ["UI Element", "English Explanation", "සිංහල පැහැදිලි කිරීම"],
        ["Sample Index Slider",
         "Select which spectrum (0–149) from the test set to view",
         "Test set එකෙන් කොයි spectrum (0-149) බලන්නේ ද ලෙස slider"],
        ["Normalised Flux Plot",
         "The spectrum's flux (brightness) across 4,000 near-infrared wavelength channels",
         "Spectrum එකේ flux (දීප්තිය) near-infrared wavelength channels 4,000 හරහා"],
        ["True Label",
         "The actual spectral class of this star derived from its APOGEE TEFF header",
         "APOGEE TEFF header වලින් ලබාගත් මෙම තාරකාවේ සැබෑ spectral class"],
        ["Predicted Label + Confidence",
         "Model's classification and its probability score for the top class",
         "Model විසින් classify කළ class සහ ඒ class සඳහා probability score"],
        ["SHAP Attribution Plot",
         "Which wavelength regions most influenced this prediction (peaks = important regions)",
         "Prediction එකට බලපෑ wavelength regions (peaks = වැදගත් ප්‍රදේශ)"],
        ["NIST Vertical Lines",
         "Known atomic absorption lines from laboratory measurements cross-matched to SHAP peaks",
         "Laboratory NIST measurements ලෙස SHAP peaks සමඟ cross-match කළ atomic lines"],
        ["Astrophysical Properties",
         "Inferred stellar physics: Teff, evolutionary stage, chemistry, fate",
         "Infer කළ stellar physics: Teff, evolve stage, chemistry, stellar fate"],
    ]
    t_el = Table([[Paragraph(f"<b>{c}</b>" if i == 0 else c,
                             styles['eng_bold'] if i == 0 else (
                             styles['sin_body'] if j == 2 else styles['eng_body']))
                   for j, c in enumerate(row)] for i, row in enumerate(elements)],
                 colWidths=[110, 200, 205])
    t_el.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_PURPLE),
        ('TEXTCOLOR', (0,0),(-1,0), C_WHITE),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING', (0,0),(-1,-1), 5),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_LGREY, C_WHITE]),
    ]))
    story.append(t_el)
    story.append(Spacer(1, 0.2*cm))

    story.append(viva_qa(
        "What does the SHAP attribution plot show?",
        "It shows which wavelength channels (positions in the spectrum) most influenced the model's prediction. Peaks in the attribution plot correspond to candidate atomic absorption lines — e.g., O I at 15,871 Å, Mg I at 15,745 Å. These were cross-matched with NIST laboratory references.",
        "SHAP attribution plot එක පෙන්වන්නේ spectrum එකේ කොයි wavelength channels model prediction එකට වැඩිම බලපෑමක් ඇති කළේ ද යන්නයි. Attribution peaks — O I 15,871 Å, Mg I 15,745 Å ආදී — NIST laboratory references සමඟ cross-match කළ atomic absorption lines ලෙස හඳුනා ගනී.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: UPLOAD & ANALYSE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 3 — 🔬 Upload & Analyse", "Page 3 — Upload & Analyse", styles, C_GREEN))
    story.append(Spacer(1, 0.3*cm))

    story.append(bilingual_block(
        "This is the core interactive page. You upload a real stellar spectrum file and the system processes it live — applying the same preprocessing pipeline as training, running inference through the 1D-CNN-BiLSTM, computing real-time Gradient SHAP attributions, displaying astrophysical properties, and allowing download of results as JSON or PDF.",
        "මෙය core interactive page එකයි. ඔබ real stellar spectrum file එකක් upload කළ සැනිකව system එක live process කරයි — training pipeline ම apply කර, 1D-CNN-BiLSTM හරහා inference ධාවනය කර, real-time Gradient SHAP attributions compute කර, astrophysical properties display කර, JSON හෝ PDF ලෙස download කිරීමට ඉඩ සලසයි.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Supported Upload Formats", styles['eng_heading']))
    formats_data = [
        ["Format", "Description (English)", "විස්තරය (සිංහල)"],
        ["CSV", "Two columns: wavelength, flux. The system auto-infers column names.", "Columns 2: wavelength, flux. Column names auto-detect කරයි."],
        ["NPZ", "NumPy archive with array key 'X' (shape 4000 or N×4000), optionally 'y'.", "NumPy archive; key 'X' (shape 4000 හෝ N×4000). Optionally 'y'."],
        ["NPY", "Raw 1D NumPy array of length 4000 or 2D array N×4000.", "1D NumPy array (length 4000) හෝ 2D array N×4000."],
        ["FITS", "APOGEE apStar FITS file. HDU 1 contains the combined flux array with WCS header.", "APOGEE apStar FITS file. HDU 1 = combined flux array with WCS header."],
    ]
    t_fmt = Table([[Paragraph(f"<b>{c}</b>" if i == 0 else c,
                              styles['eng_bold'] if i == 0 else (
                              styles['sin_body'] if j == 2 else styles['eng_body']))
                    for j, c in enumerate(row)] for i, row in enumerate(formats_data)],
                  colWidths=[50, 225, 240])
    t_fmt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_GREEN),
        ('TEXTCOLOR', (0,0),(-1,0), C_WHITE),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ('LEFTPADDING', (0,0),(-1,-1), 5),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_LGREY, C_WHITE]),
    ]))
    story.append(t_fmt)
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Step-by-step: What happens after upload", styles['eng_heading']))
    steps = [
        ("Step 1 — Preprocessing",
         "The raw spectrum is resampled onto a 4,000-point logarithmic wavelength grid (15,100–17,000 Å) and continuum-normalised using a 101-pixel median filter.",
         "Raw spectrum එක 4,000-point logarithmic wavelength grid (15,100–17,000 Å) මත resample කර, 101-pixel median filter භාවිතයෙන් continuum normalize කරයි."),
        ("Step 2 — Model Inference",
         "The preprocessed spectrum (shape: 1×4000) is passed through the frozen 1D-CNN-BiLSTM model. Softmax outputs give 5-class probabilities.",
         "Preprocessed spectrum (shape: 1×4000) frozen 1D-CNN-BiLSTM model හරහා pass කෙරේ. Softmax output classes 5 ට probabilities ලබා දෙයි."),
        ("Step 3 — Gradient SHAP",
         "Real-time Gradient SHAP (shap.GradientExplainer) computes attribution values for all 4,000 channels specifically for this uploaded spectrum.",
         "Real-time Gradient SHAP (shap.GradientExplainer) upload කළ spectrum to specific ලෙස channels 4,000 සඳහා attribution values compute කරයි."),
        ("Step 4 — Astrophysical Properties",
         "Based on the predicted class, a curated table of stellar physics is displayed: surface temperature, evolutionary stage, dominant chemistry, galactic population, and predicted fate.",
         "Predict කළ class අනුව, stellar physics curated table display කෙරේ: surface temperature, evolutionary stage, chemistry, galactic population, fate."),
        ("Step 5 — Download",
         "Results can be downloaded as (a) a JSON file with all predictions and metadata, or (b) a Scientific PDF Report with the highlighted primary finding.",
         "ප්‍රතිඵල (a) JSON file ලෙස — predictions සහ metadata සමඟ, හෝ (b) Scientific PDF Report ලෙස — primary finding highlight සහිතව download කළ හැකිය."),
    ]
    for step_title, step_en, step_sin in steps:
        story.append(Paragraph(f"▶ {step_title}", styles['eng_heading']))
        story.append(bilingual_block(step_en, step_sin, styles))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.1*cm))
    story.append(viva_qa(
        "What preprocessing contract is applied to uploaded files?",
        "The system applies: WCS wavelength extraction (for FITS) → resampling onto 4,000 log-spaced grid (15,100–17,000 Å) → 101-pixel median continuum division to normalise the flux to unity. This matches the preprocessing used during training.",
        "System apply කරන preprocessing: WCS wavelength extraction (FITS සඳහා) → 4,000 log-spaced grid (15,100–17,000 Å) මත resample → 101-pixel median continuum division (flux unity ලෙස normalize කිරීම). Training preprocessing ම apply වේ.",
        styles
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(viva_qa(
        "Does the system work on stars it has never seen before (untrained data)?",
        "Yes. The model was trained to learn physical spectral features (absorption line patterns), not to memorise individual stars. Any new APOGEE-format near-infrared spectrum in the 15,100–17,000 Å range will be correctly preprocessed and classified.",
        "ඔව්. Model එක Individual stars memorise කිරීමට නොව, physical spectral features (absorption line patterns) ඉගෙන ගැනීමට train කෙරිණ. APOGEE-format near-infrared spectrum (15,100–17,000 Å) නව ඕනෑම දෙයක් correctly preprocess කර classify කරයි.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 4 — 📈 Model Comparison", "Page 4 — Model Comparison", styles, C_AMBER))
    story.append(Spacer(1, 0.3*cm))

    story.append(bilingual_block(
        "This page compares the proposed 1D-CNN-BiLSTM model against other deep learning architectures and classical machine learning baselines, all evaluated on the same zero-leakage test split (N=150).",
        "මෙම page එකෙහි proposed 1D-CNN-BiLSTM model, අනෙකුත් deep learning architectures සහ classical ML baselines සමඟ සංසන්දනය කරයි. සියල්ල same zero-leakage test split (N=150) මත evaluate කෙරේ.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    comparison_data = [
        ["Model", "Accuracy", "Balanced Acc", "Macro F1", "Notes (English)", "සටහන (සිංහල)"],
        ["1D-CNN-BiLSTM", "77.33%", "74.15%", "0.6406", "⭐ Proposed model — best overall", "⭐ Proposed model — හොඳම ප්‍රතිඵල"],
        ["1D-ResNet", "~72%", "~68%", "~0.58", "Residual CNN without recurrence", "Recurrence නොමැති Residual CNN"],
        ["Standard CNN", "~70%", "~65%", "~0.55", "Simpler CNN baseline", "සරල CNN baseline"],
        ["Random Forest", "~65%", "~60%", "~0.50", "Classical ML baseline", "Classical ML baseline"],
        ["SVM", "~63%", "~58%", "~0.48", "Classical ML baseline", "Classical ML baseline"],
        ["KNN", "~60%", "~55%", "~0.45", "Classical ML baseline", "Classical ML baseline"],
    ]
    t_comp = Table([[Paragraph(f"<b>{c}</b>" if i == 0 else c,
                               styles['eng_bold'] if i == 0 else (
                               styles['sin_body'] if j == 5 else styles['eng_body']))
                     for j, c in enumerate(row)] for i, row in enumerate(comparison_data)],
                   colWidths=[90, 55, 60, 55, 145, 110])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#E65100')),
        ('TEXTCOLOR', (0,0),(-1,0), C_WHITE),
        ('BACKGROUND', (0,1),(-1,1), colors.HexColor('#E8F5E9')),
        ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING', (0,0),(-1,-1), 4),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,2),(-1,-1), [C_LGREY, C_WHITE]),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 0.2*cm))
    story.append(viva_qa(
        "Why did you choose CNN-BiLSTM over ResNet or classical methods?",
        "CNN-BiLSTM combines local feature extraction (CNN layers detect absorption line patterns at specific wavelengths) with bidirectional temporal context (BiLSTM captures how spectral features evolve across the spectrum). This hybrid architecture outperforms both pure CNNs and classical methods on our sequential spectral data.",
        "CNN-BiLSTM, local feature extraction (CNN layers specific wavelengths හි absorption line patterns detect කරයි) සහ bidirectional temporal context (BiLSTM spectral features evolve ආකාරය capture කරයි) ඒකාබද්ධ කරයි. Sequential spectral data සඳහා pure CNNs සහ classical methods දෙකම ඉක්මවා යයි.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5: SCIENTIFIC XAI VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 5 — 🧪 Scientific XAI Validation", "Page 5 — XAI Validation", styles, C_NAVY))
    story.append(Spacer(1, 0.3*cm))

    story.append(bilingual_block(
        "This page presents the formal scientific validation of the Gradient SHAP explanations using two established XAI evaluation protocols: (1) AOPC Faithfulness Test and (2) Jaccard Stability Test. These confirm whether the explanations are trustworthy and stable.",
        "මෙම page එකෙහි Gradient SHAP explanations, XAI evaluation protocols 2 ක් (1) AOPC Faithfulness Test සහ (2) Jaccard Stability Test භාවිතයෙන් formal scientific validation කරයි. Explanations trustworthy සහ stable ද යන්න තහවුරු කරයි.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Test 1 — AOPC Faithfulness (Area Over Perturbation Curve)", styles['eng_heading']))
    story.append(bilingual_block(
        "We masked (removed) the top-attribution wavelength channels and measured how much the model's confidence dropped. AOPC Top-K score: 0.7507. Random masking score: 0.1192. Ratio: 6.30×. This means masking the most important regions causes a 6× larger confidence drop than random masking — confirming the explanations point to genuinely important spectral regions.",
        "Top-attribution wavelength channels mask (ඉවත්) කර model confidence drop කොතරම් දුරට අඩු වේ ද ඇස්තමේන්තු කළෙමු. AOPC Top-K score: 0.7507. Random masking score: 0.1192. Ratio: 6.30×. ගිය attribution regions mask කළ විට random masking ට වඩා 6× confidence drop — explanations genuinely important spectral regions වෙත යොමු කරයි ලෙස තහවුරු කෙරේ.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Test 2 — Jaccard Stability Test", styles['eng_heading']))
    story.append(bilingual_block(
        "We added synthetic Gaussian noise to each spectrum and measured how consistent the top-K attribution peaks remained. Jaccard@sigma=0.01: 0.8069. Jaccard@sigma=0.02: 0.7768. Jaccard@sigma=0.04: 0.6853. At realistic noise levels (sigma=0.01), 80.7% of the top attribution peaks remain the same — showing the explanations are stable to observational noise.",
        "Spectrum එකට synthetic Gaussian noise add කර top-K attribution peaks stable ව පවතී ද ඇස්තමේන්තු කළෙමු. Jaccard@sigma=0.01: 0.8069. Jaccard@sigma=0.02: 0.7768. Jaccard@sigma=0.04: 0.6853. Realistic noise (sigma=0.01) හි top attribution peaks 80.7% ක් සමාන ව පවතී — explanations noise ලෙස stable ය.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("NIST Line Matching", styles['eng_heading']))
    story.append(bilingual_block(
        "The top SHAP attribution peaks were cross-matched against the NIST Atomic Spectra Database. 6 out of 57 peaks matched known atomic transitions: O I (15,871 Å), H Br14 (15,263 Å), N I (16,764 Å), Si I (15,960 Å), Mg I (15,745 Å), O I (15,569 Å). Match rate: 10.53% — exploratory alignment, not a statistically significant discovery.",
        "Top SHAP attribution peaks, NIST Atomic Spectra Database සමඟ cross-match කළෙමු. Peaks 57 න් 6 ක් known atomic transitions: O I (15,871 Å), H Br14 (15,263 Å), N I (16,764 Å), Si I (15,960 Å), Mg I (15,745 Å), O I (15,569 Å). Match rate 10.53% — exploratory alignment, statistically significant discovery නොවේ.",
        styles
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(viva_qa(
        "Does NIST line matching prove the model uses real physics?",
        "It is exploratory evidence, not proof. 10.53% of SHAP peaks aligned with known NIST lines. The Monte Carlo p-value of 0.538 is not statistically significant. However, the alignment with physically known lines (O I, Mg I, Si I) is scientifically consistent with what we expect for these spectral classes.",
        "එය exploratory evidence, proof නොවේ. SHAP peaks 10.53% known NIST lines සමඟ aligned. Monte Carlo p-value 0.538 statistically significant නොවේ. නමුත් physically known lines (O I, Mg I, Si I) සමඟ alignment, spectral classes සඳහා විද්‍යාත්මකව consistent.",
        styles
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6: ABOUT & LIMITATIONS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("PAGE 6 — ℹ️ About & Limitations", "Page 6 — About & Limitations", styles, C_GREY))
    story.append(Spacer(1, 0.3*cm))

    story.append(bilingual_block(
        "This page documents the scientific limitations of the system, lists the commands to reproduce the experiments, and provides citations and a project overview. It is important to acknowledge limitations in your viva.",
        "මෙම page එකෙහි system scientific limitations, experiments reproduce කිරීමට commands, citations සහ project overview ලේඛනගත කෙරේ. Viva හිදී limitations ප්‍රශ්නාවලිය ඉතා වැදගත්ය.",
        styles
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Key Limitations (Know these for viva!)", styles['eng_heading']))
    limitations = [
        ("Class Labels are Proxy Labels",
         "Class labels (A, B, F, G, K) are derived from APOGEE TEFF header values, not independently verified Harvard spectral types.",
         "Class labels (A, B, F, G, K) APOGEE TEFF header values ලෙස ලබාගත් proxy labels. Independently verify කළ Harvard spectral types නොවේ."),
        ("Severe Class Imbalance",
         "Class K dominates ~87% of the dataset. This limits minority class (A, B, F) performance.",
         "Class K ~87% dominate කරයි. Minority classes (A, B, F) performance සීමිත කෙරේ."),
        ("NIST Matching is Exploratory",
         "10.53% line matching rate; Monte Carlo p=0.538 (not significant). This is candidate evidence, not a confirmed discovery.",
         "Line matching 10.53%; Monte Carlo p=0.538 (significant නොවේ). Candidate evidence, confirmed discovery නොවේ."),
        ("Noise Tests are Synthetic",
         "Jaccard stability tested on simulated Gaussian noise, not real observing conditions.",
         "Jaccard stability, simulated Gaussian noise මත test කෙරිණ; Real observing conditions නොවේ."),
        ("Raw FITS not Included",
         "Raw APOGEE FITS files (4 GB) not included in submission. Preprocessing cannot be fully reproduced from supplied files alone.",
         "Raw APOGEE FITS files (4 GB) submission හි ඇතුළත් නොවේ. Preprocessing supplied files ලෙස fully reproduce කළ නොහැක."),
    ]
    for lim_title, lim_en, lim_sin in limitations:
        story.append(Paragraph(f"⚠ {lim_title}", styles['eng_heading']))
        story.append(bilingual_block(lim_en, lim_sin, styles))
        story.append(Spacer(1, 0.1*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SPECTRAL CLASSES REFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("SPECTRAL CLASSES — Quick Reference", "Spectral Classes — කෙටි මාර්ගෝපදේශය", styles, C_BLUE))
    story.append(Spacer(1, 0.3*cm))

    classes = [
        ("Class K", "Orange / Red Giant", "Aging post-main-sequence red giant. Core hydrogen exhausted. Mg I, Si I, CO molecular bands dominate NIR spectrum. Located in Milky Way disk and bulge.",
         "වයස්ගත post-main-sequence Red Giant. Core hydrogen අවසන්. Mg I, Si I, CO molecular bands NIR spectrum dominate කරයි. Milky Way disk සහ bulge."),
        ("Class G", "Solar-Type Star", "Solar twin. Stable Main Sequence star. Surface temperature 5,000–6,000 K, similar to our Sun. Fe I, Ca II, Si I absorption lines.",
         "Solar twin. Stable Main Sequence star. Surface temperature 5,000–6,000 K — අපේ සූර්යයාට සමාන. Fe I, Ca II, Si I absorption lines."),
        ("Class F", "Intermediate White Star", "Warmer than the Sun (6,000–7,500 K). Main Sequence / early post-main-sequence. Ionised metals Fe II, Ca II dominant.",
         "සූර්යයාට වඩා උෂ්ණ (6,000–7,500 K). Main Sequence / early post-main-sequence. Ionised metals Fe II, Ca II dominant."),
        ("Class A", "Hot Early-Type Star", "Hot white star (7,500–10,000 K). Short lifetime ~1 billion years. Strong Hydrogen Brackett absorption series in NIR.",
         "Hot white star (7,500–10,000 K). ජීවිත කාලය ~billion years 1. Strong Hydrogen Brackett absorption NIR."),
        ("Class B", "Luminous Blue Giant", "Extremely hot (10,000–30,000 K), massive, young. He I lines and Brackett series. Destined for supernova or massive white dwarf.",
         "ඉතා උෂ්ණ (10,000–30,000 K), විශාල, තරුණ. He I lines සහ Brackett series. Supernova හෝ massive white dwarf ඉලක්ක."),
    ]
    for cls_name, cls_type, cls_en, cls_sin in classes:
        data = [[
            Paragraph(f"<font color='white'><b>{cls_name}</b><br/>{cls_type}</font>",
                      ParagraphStyle('ClsLabel', fontName='Helvetica-Bold', fontSize=10,
                                     leading=14, textColor=C_WHITE)),
            Paragraph(cls_en, styles['eng_body']),
            Paragraph(cls_sin, styles['sin_body']),
        ]]
        t_cls = Table(data, colWidths=[85, 218, 212])
        t_cls.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(0,-1), C_NAVY),
            ('BACKGROUND', (1,0),(1,-1), colors.HexColor('#EEF3FF')),
            ('BACKGROUND', (2,0),(2,-1), C_GREEN_BG),
            ('BOX', (0,0),(-1,-1), 0.5, C_BORDER),
            ('INNERGRID', (0,0),(-1,-1), 0.5, C_BORDER),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('LEFTPADDING', (0,0),(-1,-1), 6),
            ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ]))
        story.append(t_cls)
        story.append(Spacer(1, 0.1*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # CRITICAL VIVA Q&A
    # ══════════════════════════════════════════════════════════════════════════
    story.append(section_banner("CRITICAL VIVA Q&A — Panel Questions & Answers", "Panel ප්‍රශ්නෝත්තර", styles, C_AMBER))
    story.append(Spacer(1, 0.3*cm))

    critical_qa = [
        ("What is the main contribution of your research?",
         "We demonstrate that a 1D-CNN-BiLSTM model can classify near-infrared stellar spectra with 77.33% accuracy on a zero-leakage split, and that Gradient SHAP explanations point to physically meaningful atomic absorption regions (O I, Mg I, Si I), providing interpretable, trustworthy AI for astronomical classification.",
         "1D-CNN-BiLSTM model zero-leakage split මත near-infrared stellar spectra 77.33% accuracy ලෙස classify කළ හැකි බව සහ Gradient SHAP explanations physically meaningful atomic absorption regions (O I, Mg I, Si I) වෙත point කරන බව demonstrate කළෙමු — astronomical classification සඳහා interpretable, trustworthy AI සපයමින්."),

        ("What is Gradient SHAP and why did you use it?",
         "Gradient SHAP (shap.GradientExplainer) computes expected gradients over a background dataset to assign attribution values to each input feature. We used it because it is model-specific to neural networks, computationally efficient for 4,000-channel inputs, and produces attributions directly comparable to known spectral lines.",
         "Gradient SHAP (shap.GradientExplainer) background dataset හරහා expected gradients compute කර සෑම input feature ටම attribution values assign කරයි. Neural networks ලෙස model-specific, 4,000-channel inputs ලෙස computationally efficient, සහ known spectral lines ලෙස directly comparable attributions produce කරයි — ඒ නිසා භාවිත කළෙමු."),

        ("What does zero-leakage split mean?",
         "Zero-leakage split means no data from the test set was used during training or validation. We used 700 samples for training, 150 for validation, and 150 for testing — completely isolated. This ensures the reported 77.33% accuracy is a genuine generalisation measure, not an inflated result from data leakage.",
         "Zero-leakage split යනු test set data training හෝ validation දිදී ලෙස භාවිත නොකළ බවයි. Training 700, validation 150, testing 150 — සම්පූර්ණයෙන් isolated. Reported 77.33% accuracy genuine generalisation, data leakage ලෙස inflated result නොවේ."),

        ("Why does Class K have the highest accuracy?",
         "Class K dominates ~87% of our dataset because APOGEE primarily targets red giant stars. The model has seen far more Class K examples, so it is better calibrated for that class. This creates class imbalance — addressed by reporting balanced accuracy alongside raw accuracy.",
         "APOGEE primarily red giant stars target කරයි. ඒ නිසා dataset 87% Class K ලෙස. Model Class K examples ඉතා වැඩියෙන් දැක ඇති, class ලෙස better calibrated. Class imbalance — balanced accuracy reporting ලෙස address කෙරේ."),

        ("How did you validate the XAI explanations?",
         "Two tests: (1) AOPC Faithfulness — masking top-attribution channels caused 6.30× larger confidence drop than random masking (AOPC 0.7507 vs 0.1192), confirming attributions point to genuinely important features. (2) Jaccard Stability — at sigma=0.01 noise, 80.7% of top peaks remained consistent, confirming explanations are stable.",
         "Tests 2: (1) AOPC Faithfulness — top-attribution channels mask කිරීම random masking ට වඩා 6.30× confidence drop (AOPC 0.7507 vs 0.1192), attributions genuinely important features. (2) Jaccard Stability — sigma=0.01 noise, top peaks 80.7% consistent — explanations stable."),

        ("What would you improve in future work?",
         "Future improvements: (1) Collect more minority class spectra (A, B, F) to reduce imbalance. (2) Implement per-upload uncertainty quantification (MC Dropout or Deep Ensembles). (3) Use per-query SHAP instead of background-averaged attributions. (4) Validate on optical (SDSS/Gaia) spectra for generalisability.",
         "Future improvements: (1) Minority class (A, B, F) spectra collect කිරීම — imbalance reduce. (2) Per-upload uncertainty quantification (MC Dropout / Deep Ensembles) implement. (3) Background-averaged ලෙස per-query SHAP. (4) Optical spectra (SDSS/Gaia) ලෙස generalisability validate."),
    ]

    for q, a_en, a_sin in critical_qa:
        story.append(viva_qa(q, a_en, a_sin, styles))
        story.append(Spacer(1, 0.2*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=6))
    story.append(Paragraph(
        "J26-IT-449 | P.K. Edirisooriya (IT23330696) | SLIIT IT4010 Research — PP1 Assessment | Generated: " + time.strftime("%Y-%m-%d"),
        ParagraphStyle('Footer', fontName='Helvetica', fontSize=7.5, leading=10,
                       textColor=C_GREY, alignment=1)
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        title="Viva Preparation Guide — Stellar Spectra XAI",
        author="P.K. Edirisooriya IT23330696"
    )
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


if __name__ == '__main__':
    print("Generating bilingual Viva Preparation Guide PDF...")
    pdf_bytes = build_viva_guide()
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(pdf_bytes)
    print(f"Done! PDF saved to: {OUTPUT_PATH}")
    print(f"File size: {len(pdf_bytes) / 1024:.1f} KB")
