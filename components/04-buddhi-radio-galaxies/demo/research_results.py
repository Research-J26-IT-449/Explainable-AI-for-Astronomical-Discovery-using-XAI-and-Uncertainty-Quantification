"""Read-only, aggregate research presentation; no dataset/model imports."""
import hashlib
from html import escape
import json
import os
from pathlib import Path

ASSETS=Path(__file__).resolve().parent / "assets/public"


def public_mode():
    """Public presentation is the default; the deployment entry enforces it."""
    return os.environ.get("RADIOMORPH_PUBLIC_DEMO", "1") != "0"


def results():
    provenance=json.loads((ASSETS/"provenance.json").read_bytes())
    for name,digest in provenance["public_files"].items():
        if name not in {"research_results.json","confusion_matrix.png"}:
            raise ValueError("Unexpected public research asset reference.")
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Public research asset checksum mismatch.")
    return json.loads((ASSETS/"research_results.json").read_bytes())


def research_html():
    values=results()
    rows="".join(f"<tr><td>{escape(name)}</td><td>{score:.6f}</td></tr>" for name,score in values["aopc30"].items())
    return ('<div class="rm-section-heading"><span class="rm-eyebrow">PRECOMPUTED / DEVELOPMENT RESULTS</span>'
        '<h2>Verified research results</h2><p>Saved corrected-model results. These are not predictions for a visitor upload.</p></div>'
        '<div class="rm-info-grid"><article class="rm-info-card"><h3>Corrected validation</h3>'
        f'<p><b>{values["accuracy"]:.6%}</b> accuracy · <b>{values["macro_f1"]:.6f}</b> macro F1</p>'
        f'<p>{values["validation_count"]} validation images; FR-I support 51, FR-II support 65. '
        'The checkpoint was selected on this cohort; independent TEST and external generalization are not established.</p></article>'
        '<article class="rm-info-card"><h3>AOPC30 comparison</h3><table><caption>Mean fixed-class probability drop, deletion through 30%</caption>'
        '<thead><tr><th>Ranking</th><th>Mean AOPC30</th></tr></thead><tbody>'+rows+'</tbody></table>'
        '<p>Border-median replacement; 20 random controls per image. Eight negative CAM AOPCs were retained. '
        'Deletion can create out-of-distribution inputs; this does not prove physical correctness.</p></article>'
        '<article class="rm-info-card"><h3>116 genuine Grad-CAM explanations</h3>'
        '<p>Predicted-class logit attribution at blocks.2.convolution.3; native 19×19, aligned 150×150 display.</p>'
        '<p>Noise tests showed class flips, clipping and degenerate transformed maps. Expert-mask physics alignment '
        'is unavailable, and parameter-randomization checks remain deferred.</p></article></div>')


def local_examples():
    """Private optional gallery; never enabled in the deployable public entry."""
    if public_mode() or os.environ.get("RADIOMORPH_SHOW_LOCAL_EXAMPLES")!="1":
        return []
    root=Path(__file__).resolve().parents[1]
    directory=root/"analysis_reports/step21_corrected_gradcam_evaluation_attempt02"
    ledger=json.loads((directory/"final_report_manifest.json").read_bytes())["sha256"]
    selected=[(35,"Correct FR-I"),(31,"Incorrect FR-I; predicted FR-II"),
              (50,"Correct FR-II"),(437,"Incorrect FR-II; predicted FR-I")]
    items=[]
    for index,label in selected:
        reference=f"representatives/index_{index}.png"
        path=directory/reference
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ledger[reference]:
            raise ValueError("Precomputed local example checksum mismatch.")
        items.append((str(path),f"Index {index}: {label}. Exploratory; not expert-validated."))
    return items
