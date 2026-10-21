import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_document():
    doc = docx.Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("RESEARCH METHODOLOGY SPECIFICATION\n")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 121)
    
    sub_title = title_p.add_run("Module 1: SHAP-Driven Feature Discovery in Stellar Spectra\n")
    sub_title.font.name = 'Calibri'
    sub_title.font.size = Pt(14)
    sub_title.font.italic = True
    
    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.add_run("Project ID: J26-IT-449 | Researcher: P.K. Edirisooriya (IT23330696)\nCourse: IT4010 Research Project | Assessment: Progress Presentation 1 (PP1)\nSLIIT School of Computing\n").font.size = Pt(10)
    
    doc.add_heading("1. Introduction & Research Problem", level=1)
    p = doc.add_paragraph(
        "Modern wide-field spectroscopic sky surveys, such as SDSS-IV APOGEE, acquire hundreds of thousands of "
        "high-resolution stellar spectra across the near-infrared passband. While modern deep convolutional neural networks "
        "excel at classifying spectral features and inferring atmospheric parameters, they operate as opaque 'black boxes'. "
        "In astrophysics, an accurate prediction is scientifically insufficient if the underlying physical justification "
        "remains unexplained. Astrophysicists must understand whether neural networks learn genuine physical atomic and "
        "molecular transitions (e.g., hydrogen Brackett series, iron, silicon, magnesium lines) or exploit unphysical continuum "
        "slopes, sky subtraction residuals, and instrument artifacts."
    )
    
    doc.add_heading("2. Research Gap & Proposed Novelty", level=1)
    doc.add_paragraph(
        "Existing astronomical literature predominantly focuses on raw classification accuracy or applies post-hoc saliency maps "
        "without quantitative physical validation. The research gap addressed by this study is twofold:\n"
        "1. The lack of rigorous, baseline-controlled quantitative alignment between Explainable AI (XAI) attributions and NIST laboratory atomic transitions.\n"
        "2. The absence of systematic evaluation of XAI faithfulness (AOPC) and stability under realistic observational survey noise."
    )
    
    doc.add_heading("3. Observational Dataset & Preprocessing", level=1)
    doc.add_paragraph(
        "The primary dataset comprises 1,000 genuine near-infrared spectra from the SDSS-IV APOGEE DR17 survey (H-band: 15,100 to 17,000 Å, "
        "R ~ 22,500). Raw FITS spectra from the APO 2.5m and LCO 2.5m telescopes are resampled onto a standardized 4,000-point logarithmic "
        "wavelength grid (dispersion ~0.475 Å/pixel) and continuum-normalized using a 101-pixel median filter. "
        "Classes (K, G, F, A, B) represent effective-temperature proxy categories derived from ASPCAP TEFF header measurements."
    )
    
    doc.add_heading("4. Deep Learning Model Architectures", level=1)
    doc.add_paragraph(
        "Three distinct deep learning paradigms are evaluated:\n"
        "1. 1D-CNN-BiLSTM (Hybrid): Combines spatial 1D convolutional feature extraction (kernels 11, 7) with a 2-layer Bidirectional LSTM "
        "to capture long-range pseudo-continuum and multi-element ionization balances.\n"
        "2. 1D-ResNet: Deep residual architecture utilizing skip connections across 3 residual blocks.\n"
        "3. Standard 1D-CNN: Baseline convolutional neural network with max pooling and dropout."
    )
    
    doc.add_heading("5. Explainable AI & Scientific Validation Framework", level=1)
    doc.add_paragraph(
        "1. DeepSHAP Attribution: Implemented via shap.GradientExplainer over the trained CNN-BiLSTM, producing attribution tensors of shape (N, 4000, 5).\n"
        "2. Peak Detection & NIST Matching: Attribution peaks are detected using an 85th percentile prominence threshold and matched against 31 "
        "known near-infrared transitions within a +/-3.5 Å astrophysical window.\n"
        "3. Monte Carlo Random Baseline: Compares observed spectral line alignment against 1,000 randomized synthetic attribution trials to establish empirical p-values.\n"
        "4. Faithfulness (AOPC): Measures target confidence degradation when sequentially masking top-ranked SHAP channels vs random channels.\n"
        "5. Stability (Jaccard): Evaluates explanation region consistency under simulated observational noise (SNR = 100, 50, 25)."
    )
    
    doc.add_heading("6. Key Empirical Results Summary", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    hdr[0].text = "Model / Evaluation Metric"
    hdr[1].text = "Zero-Leakage Clean Split"
    hdr[2].text = "Statistical Baseline"
    hdr[3].text = "Scientific Significance"
    
    rows_data = [
        ("1D-CNN-BiLSTM Test Accuracy", "77.33% (116/150)", "64.67% (Majority Class)", "Exceeds zero-rule baseline"),
        ("1D-CNN-BiLSTM Balanced Accuracy", "74.15%", "20.00% (Majority Class)", "Highest balance across classes"),
        ("Spectral Line Alignment Score", "10.53% (6/57 peaks)", "10.19% +/- 3.94%", "p = 0.5380 (diffuse learning)"),
        ("XAI Faithfulness Ratio (AOPC)", "6.30x vs Random", "1.00x (Baseline)", "p < 0.001 (High Faithfulness)"),
        ("XAI Stability (SNR=50)", "Jaccard = 0.7768", "N/A", "High observational stability")
    ]
    for r in rows_data:
        row = table.add_row().cells
        row[0].text = r[0]
        row[1].text = r[1]
        row[2].text = r[2]
        row[3].text = r[3]
        
    doc.add_heading("7. Disclosures & Scientific Limitations", level=1)
    doc.add_paragraph(
        "1. Class Imbalance: Classes K and G represent 89.8% of the dataset due to APOGEE targeting of red giants.\n"
        "2. Proxy Classification: Target classes are effective-temperature proxies, not true optical MK classifications.\n"
        "3. Diffuse Spectral Learning: The model utilizes broader molecular bands and continuum slopes alongside atomic lines."
    )
    
    doc.save("docs/STELLAR_SPECTRA_RESEARCH_METHODOLOGY.docx")
    print("Saved docs/STELLAR_SPECTRA_RESEARCH_METHODOLOGY.docx")

if __name__ == '__main__':
    create_document()
