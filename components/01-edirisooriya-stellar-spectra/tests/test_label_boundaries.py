import os, pandas as pd, json, numpy as np
from scipy.io import readsav

# Test that every label in the manifest matches the temperature-bin mapping defined in query_apogee_labels.py

def teff_to_class(teff):
    TEMP_BINS = [0, 3500, 5000, 6000, 7500, 10000, 30000, 50000]
    CLASS_NAMES = ['M', 'K', 'G', 'F', 'A', 'B', 'O']
    for i in range(len(TEMP_BINS) - 1):
        if TEMP_BINS[i] <= teff < TEMP_BINS[i + 1]:
            return CLASS_NAMES[i]
    return 'UNKNOWN'

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
manifest_path = os.path.join(ROOT, 'dataset_manifest_labeled.csv')
manifest = pd.read_csv(manifest_path)

def test_label_consistency():
    for _, row in manifest.iterrows():
        expected = teff_to_class(row['teff'])
        assert expected == row['label'], f"Row {row['file_name']} TEFF={row['teff']} gave label {row['label']}, expected {expected}"
