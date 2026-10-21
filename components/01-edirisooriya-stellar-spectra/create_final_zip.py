"""
create_final_zip.py
===================
Creates MODULE1_STELLAR_SPECTRA_PP1_FINAL_VERIFIED.zip
Includes all essential project files but excludes large raw FITS data and old validation zip.
"""
import os
import zipfile

ROOT = os.path.abspath("module1_stellar_spectra")
OUT_ZIP = os.path.join(ROOT, "MODULE1_STELLAR_SPECTRA_PP1_FINAL_VERIFIED.zip")

# Directories to include (relative to ROOT)
INCLUDE_DIRS = [
    "reports",
    "docs",
    "presentation",
    "results",
    "models/corrected_split",
    "processed/corrected_clean",
    "tests",
    "plots",
]

# Individual top-level files to include (relative to ROOT)
INCLUDE_FILES = [
    "app.py",
    "model_architectures.py",
    "01_data_preprocess.py",
    "03_generate_clean_shap.py",
    "04_scientific_validation.py",
    "05_xai_faithfulness.py",
    "06_xai_stability.py",
    "07_run_baselines.py",
    "train_corrected_split.py",
    "requirements.txt",
    "run_app.bat",
    "run_app.sh",
    "README.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "MODEL_CARD.md",
    "DATASET_PROVENANCE.md",
    "dataset_manifest_labeled.csv",
    "metrics.json",
    "line_matches.csv",
    "confusion_matrix.txt",
    "shap_attributions.npz",
    "three_models_benchmark_results.json",
    "generate_final_reports_and_verifications.py",
    "generate_final_pp1_readiness.py",
]

# File extensions to exclude from directory sweeps (large/temp files)
EXCLUDE_EXTENSIONS = {".fits", ".zip", ".tar", ".gz", ".log", ".pyc"}
EXCLUDE_DIRS = {"venv", "__pycache__", ".pytest_cache", "raw_spectra"}
EXCLUDE_FILENAMES = {
    "RAW_APOGEE_FITS_DATASET.zip",
    "MODULE1_STELLAR_SPECTRA_CORRECTED_AUDIT.zip",
    "MODULE1_STELLAR_SPECTRA_PP1_VALIDATED.zip",
    "MODULE1_STELLAR_SPECTRA_PP1_FINAL_VERIFIED.zip",
    "download_log.txt",
    "inventory.txt",
    "apo25m_index.html",
}

added = 0
skipped = 0

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    # Add specific top-level files
    for fname in INCLUDE_FILES:
        fpath = os.path.join(ROOT, fname)
        if os.path.exists(fpath):
            arcname = os.path.join("module1_stellar_spectra", fname)
            zf.write(fpath, arcname)
            added += 1
        else:
            print(f"  [SKIP] Not found: {fname}")
            skipped += 1

    # Add full directory trees
    for rel_dir in INCLUDE_DIRS:
        abs_dir = os.path.join(ROOT, rel_dir)
        if not os.path.exists(abs_dir):
            print(f"  [SKIP] Directory not found: {rel_dir}")
            skipped += 1
            continue
        for dirpath, dirnames, filenames in os.walk(abs_dir):
            # Skip excluded subdirectories
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                if fname in EXCLUDE_FILENAMES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue
                fpath = os.path.join(dirpath, fname)
                # File size guard: skip anything > 50 MB
                try:
                    fsize_mb = os.path.getsize(fpath) / (1024**2)
                    if fsize_mb > 50:
                        print(f"  [SKIP] Too large ({fsize_mb:.1f}MB): {fpath}")
                        skipped += 1
                        continue
                except OSError:
                    continue
                rel_path = os.path.relpath(fpath, os.path.dirname(ROOT))
                zf.write(fpath, rel_path)
                added += 1

zip_size_mb = os.path.getsize(OUT_ZIP) / (1024**2)
print(f"\n{'='*60}")
print(f"FINAL ZIP: {os.path.basename(OUT_ZIP)}")
print(f"Location:  {OUT_ZIP}")
print(f"Size:      {zip_size_mb:.2f} MB")
print(f"Files:     {added} added, {skipped} skipped")
print(f"{'='*60}")
