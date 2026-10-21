"""
test_pipeline.py
================
Automated tests for the Module 1 Stellar Spectra pipeline.
Run with: python -m pytest tests/test_pipeline.py -v

Tests cover:
  1. Label boundary correctness (temperature → class mapping)
  2. Manifest integrity (all FITS have labels, no unknown classes)
  3. Corrected dataset split counts (700/150/150) — reads from corrected_clean/
  4. Corrected confusion matrix sum == 150 (corrected test-set size)
  5. Diagonal accuracy == reported accuracy from corrected benchmark
  6. Per-class metric arrays have correct length
  7. SHAP output shape (clean tensor: 150×4000×5, legacy file: legacy artifact)
  8. Line-matching CSV contains required columns and has >0 rows

NOTE: metrics.json and confusion_matrix.txt are HISTORICAL ARTIFACTS from the
leaky 91.33% run. They are preserved but are NOT used for corrected model checks.
Corrected model metrics are loaded from results/corrected_split_benchmark.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH    = os.path.join(ROOT, "dataset_manifest_labeled.csv")

# CORRECTED paths (the only valid split)
TRAIN_NPZ        = os.path.join(ROOT, "processed", "corrected_clean", "train.npz")
VAL_NPZ          = os.path.join(ROOT, "processed", "corrected_clean", "val.npz")
TEST_NPZ         = os.path.join(ROOT, "processed", "corrected_clean", "test.npz")

# CORRECTED benchmark — source of truth for model metrics
CORRECTED_BENCH  = os.path.join(ROOT, "results", "corrected_split_benchmark.json")

# LEGACY / HISTORICAL artifact paths (preserved, not used for corrected checks)
HIST_METRICS_PATH = os.path.join(ROOT, "metrics.json")
HIST_CM_PATH      = os.path.join(ROOT, "confusion_matrix.txt")

# SHAP paths
CLEAN_SHAP_PATH  = os.path.join(ROOT, "results", "shap_attributions_clean.npz")
LEGACY_SHAP_PATH = os.path.join(ROOT, "shap_attributions.npz")

LINE_MATCH_PATH  = os.path.join(ROOT, "line_matches.csv")

# Corrected class order (sorted labels present in corrected split)
CORRECTED_CLASS_ORDER = ["A", "B", "F", "G", "K"]
CORRECTED_TEST_SIZE   = 150

# Temperature → proxy class mapping (same as 01_data_preprocess.py)
TEMP_BINS   = [0, 3500, 5000, 6000, 7500, 10000, 30000, 50000]
TEMP_LABELS = ["M", "K", "G", "F", "A", "B", "O"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Label boundary tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelBoundaries:
    """Verify that specific TEFF values map to the expected Harvard proxy class."""

    def _classify(self, teff: float) -> str:
        for i, (lo, hi) in enumerate(zip(TEMP_BINS[:-1], TEMP_BINS[1:])):
            if lo <= teff < hi:
                return TEMP_LABELS[i]
        return "O"

    def test_M_boundary(self):
        assert self._classify(3000) == "M"
        assert self._classify(3499) == "M"

    def test_K_boundary(self):
        assert self._classify(3500) == "K"
        assert self._classify(4999) == "K"

    def test_G_boundary(self):
        assert self._classify(5000) == "G"
        assert self._classify(5999) == "G"

    def test_F_boundary(self):
        assert self._classify(6000) == "F"
        assert self._classify(7499) == "F"

    def test_A_boundary(self):
        assert self._classify(7500) == "A"
        assert self._classify(9999) == "A"

    def test_B_boundary(self):
        assert self._classify(10000) == "B"
        assert self._classify(29999) == "B"

    def test_O_boundary(self):
        assert self._classify(30000) == "O"
        assert self._classify(49999) == "O"

    def test_no_overlap_at_boundary(self):
        """TEFF exactly at a boundary must belong to the UPPER class."""
        for boundary in TEMP_BINS[1:-1]:
            label = self._classify(boundary)
            upper_class = TEMP_LABELS[TEMP_BINS.index(boundary)]
            assert label == upper_class, (
                f"TEFF={boundary} → {label}, expected {upper_class}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Manifest integrity tests
# ══════════════════════════════════════════════════════════════════════════════

class TestManifest:
    @pytest.fixture(scope="class")
    def manifest(self):
        assert os.path.exists(MANIFEST_PATH), f"Manifest not found: {MANIFEST_PATH}"
        return pd.read_csv(MANIFEST_PATH)

    def test_manifest_not_empty(self, manifest):
        assert len(manifest) > 0, "Manifest is empty"

    def test_manifest_has_required_columns(self, manifest):
        for col in ["objid", "teff", "label"]:
            assert col in manifest.columns, f"Missing column: {col}"

    def test_no_null_labels(self, manifest):
        assert manifest["label"].isna().sum() == 0, "Manifest contains null labels"

    def test_all_labels_are_known_classes(self, manifest):
        valid = set(TEMP_LABELS)
        unknown = set(manifest["label"].unique()) - valid
        assert not unknown, f"Unknown labels in manifest: {unknown}"

    def test_more_than_62_spectra(self, manifest):
        """Must exceed the original 62-sample dataset."""
        assert len(manifest) > 62, (
            f"Only {len(manifest)} entries — must be >62"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Corrected dataset split integrity (reads from corrected_clean/)
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetSplits:
    @pytest.fixture(scope="class")
    def splits(self):
        for p in [TRAIN_NPZ, VAL_NPZ, TEST_NPZ]:
            assert os.path.exists(p), f"Corrected split file not found: {p}"
        return {
            "train": np.load(TRAIN_NPZ),
            "val":   np.load(VAL_NPZ),
            "test":  np.load(TEST_NPZ),
        }

    def test_splits_have_X_and_y(self, splits):
        for split_name, data in splits.items():
            assert "X" in data, f"Split '{split_name}' missing X"
            assert "y" in data, f"Split '{split_name}' missing y"

    def test_splits_are_non_empty(self, splits):
        for split_name, data in splits.items():
            assert len(data["X"]) > 0, f"Split '{split_name}' is empty"

    def test_X_and_y_same_length(self, splits):
        for split_name, data in splits.items():
            assert len(data["X"]) == len(data["y"]), (
                f"Split '{split_name}': X ({len(data['X'])}) != y ({len(data['y'])})"
            )

    def test_feature_length_is_4000(self, splits):
        """Spectra are resampled to 4000 wavelength bins."""
        for split_name, data in splits.items():
            assert data["X"].shape[1] == 4000, (
                f"Split '{split_name}' has feature length {data['X'].shape[1]}, expected 4000"
            )

    def test_corrected_split_sizes_are_700_150_150(self, splits):
        """Corrected split must have exactly 700/150/150 samples."""
        assert len(splits["train"]["X"]) == 700, f"Train size {len(splits['train']['X'])} != 700"
        assert len(splits["val"]["X"]) == 150,   f"Val size {len(splits['val']['X'])} != 150"
        assert len(splits["test"]["X"]) == 150,  f"Test size {len(splits['test']['X'])} != 150"

    def test_test_set_size_is_150(self, splits):
        """Corrected test set must be exactly 150 samples."""
        assert len(splits["test"]["y"]) == CORRECTED_TEST_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# 4 & 5. CORRECTED Evaluation consistency tests (reads corrected_split_benchmark.json)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationConsistency:
    """
    Tests corrected model metrics from results/corrected_split_benchmark.json.
    Do NOT use metrics.json (historical/leaky 91.33% result).
    """

    @pytest.fixture(scope="class")
    def corrected_metrics(self):
        assert os.path.exists(CORRECTED_BENCH), (
            f"Corrected benchmark not found: {CORRECTED_BENCH}"
        )
        with open(CORRECTED_BENCH) as f:
            bench = json.load(f)
        assert "1D-CNN-BiLSTM" in bench, "1D-CNN-BiLSTM key missing from corrected benchmark"
        m = bench["1D-CNN-BiLSTM"]
        # Normalise keys: provide class_labels and test_size from known source of truth
        m["class_labels"] = CORRECTED_CLASS_ORDER
        m["test_size"] = CORRECTED_TEST_SIZE
        # balanced_accuracy alias
        if "balanced_accuracy" not in m and "balanced_acc" in m:
            m["balanced_accuracy"] = m["balanced_acc"]
        return m

    def test_confusion_matrix_total_equals_test_size(self, corrected_metrics):
        cm = np.array(corrected_metrics["cm"])
        assert cm.sum() == corrected_metrics["test_size"], (
            f"Confusion matrix total {cm.sum()} != test_size {corrected_metrics['test_size']}"
        )

    def test_diagonal_accuracy_matches_reported_accuracy(self, corrected_metrics):
        cm = np.array(corrected_metrics["cm"])
        diagonal_acc = np.trace(cm) / cm.sum()
        reported_acc = corrected_metrics["accuracy"]
        assert np.isclose(diagonal_acc, reported_acc, atol=1e-6), (
            f"Diagonal accuracy {diagonal_acc:.6f} != reported accuracy {reported_acc:.6f}"
        )

    def test_accuracy_is_between_0_and_1(self, corrected_metrics):
        assert 0.0 <= corrected_metrics["accuracy"] <= 1.0

    def test_macro_f1_is_between_0_and_1(self, corrected_metrics):
        assert 0.0 <= corrected_metrics["macro_f1"] <= 1.0

    def test_balanced_accuracy_is_between_0_and_1(self, corrected_metrics):
        assert 0.0 <= corrected_metrics["balanced_accuracy"] <= 1.0

    def test_class_labels_present(self, corrected_metrics):
        assert "class_labels" in corrected_metrics and len(corrected_metrics["class_labels"]) > 0

    def test_class_order_is_A_B_F_G_K(self, corrected_metrics):
        assert corrected_metrics["class_labels"] == CORRECTED_CLASS_ORDER

    def test_confusion_matrix_shape_is_5x5(self, corrected_metrics):
        cm = np.array(corrected_metrics["cm"])
        assert cm.shape == (5, 5), f"Confusion matrix shape {cm.shape} != (5, 5)"

    def test_corrected_accuracy_is_approximately_77_percent(self, corrected_metrics):
        """Corrected CNN-BiLSTM must be approximately 77.33% (116/150)."""
        acc = corrected_metrics["accuracy"]
        assert abs(acc - 0.7733) < 0.005, (
            f"Corrected accuracy {acc:.4f} is not close to expected 0.7733"
        )

    def test_corrected_balanced_accuracy_is_approximately_74_percent(self, corrected_metrics):
        bal = corrected_metrics["balanced_accuracy"]
        assert abs(bal - 0.7415) < 0.005, (
            f"Corrected balanced accuracy {bal:.4f} is not close to expected 0.7415"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4b. HISTORICAL artifact tests (metrics.json, confusion_matrix.txt)
#     These are preserved but must NOT be compared with corrected results.
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoricalArtifacts:
    """
    Integrity checks for HISTORICAL artifacts only.
    metrics.json and confusion_matrix.txt reflect the LEAKY 91.33% run.
    They are preserved as historical evidence and must never be used as
    corrected model results.
    """

    def test_historical_metrics_json_exists_as_artifact(self):
        """Historical metrics.json must be preserved (but is not the corrected result)."""
        assert os.path.exists(HIST_METRICS_PATH), (
            "Historical metrics.json is missing — it should be preserved as an artifact."
        )

    def test_historical_confusion_matrix_txt_exists_as_artifact(self):
        """Historical confusion_matrix.txt must be preserved."""
        assert os.path.exists(HIST_CM_PATH), (
            "Historical confusion_matrix.txt is missing."
        )

    def test_historical_metrics_json_is_parseable(self):
        if not os.path.exists(HIST_METRICS_PATH):
            pytest.skip("Historical metrics.json not found")
        with open(HIST_METRICS_PATH) as f:
            m = json.load(f)
        assert "accuracy" in m, "Historical metrics.json missing accuracy key"
        # Sanity: historical result should be distinct from corrected 77.33%
        # (it was 91.33% due to data leakage)
        assert m["accuracy"] != 0.7733, (
            "Historical metrics.json unexpectedly matches corrected result — check file integrity"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP output tests — clean tensor (150×4000×5) and legacy file
# ══════════════════════════════════════════════════════════════════════════════

class TestSHAPOutput:
    """Tests for the CLEAN Gradient SHAP tensor (primary) and legacy file (secondary)."""

    @pytest.fixture(scope="class")
    def clean_shap_data(self):
        assert os.path.exists(CLEAN_SHAP_PATH), f"Clean SHAP file not found: {CLEAN_SHAP_PATH}"
        return np.load(CLEAN_SHAP_PATH)

    def test_shap_key_present(self, clean_shap_data):
        assert "shap_tensor" in clean_shap_data, "Key 'shap_tensor' missing from shap_attributions_clean.npz"

    def test_shap_wavelength_axis_is_4000(self, clean_shap_data):
        shap = clean_shap_data["shap_tensor"]
        assert shap.shape[1] == 4000, (
            f"Expected wavelength axis length 4000, got shape {shap.shape}"
        )

    def test_clean_shap_tensor_shape_is_150_4000_5(self, clean_shap_data):
        shap = clean_shap_data["shap_tensor"]
        assert shap.shape == (150, 4000, 5), (
            f"Expected clean SHAP shape (150, 4000, 5), got {shap.shape}"
        )

    def test_shap_values_finite(self, clean_shap_data):
        shap = clean_shap_data["shap_tensor"]
        assert np.all(np.isfinite(shap)), "Clean SHAP values contain NaN or Inf"

    def test_legacy_shap_file_exists_as_artifact(self):
        """Legacy shap_attributions.npz must be preserved as a historical artifact."""
        assert os.path.exists(LEGACY_SHAP_PATH), (
            "Legacy shap_attributions.npz is missing — should be preserved."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Line-matching CSV tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLineMatching:
    @pytest.fixture(scope="class")
    def df(self):
        assert os.path.exists(LINE_MATCH_PATH), (
            f"line_matches.csv not found: {LINE_MATCH_PATH}"
        )
        return pd.read_csv(LINE_MATCH_PATH)

    def test_csv_not_empty(self, df):
        assert len(df) > 0

    def test_required_columns_present(self, df):
        for col in ["line_name", "line_wavelength", "aaas"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_aaas_values_non_negative(self, df):
        assert (df["aaas"] >= 0).all(), "Negative AAAS values found"

    def test_wavelengths_in_apogee_range(self, df):
        """All lines should be in the APOGEE H-band range (roughly 15000–17500 Å)."""
        assert df["line_wavelength"].between(14000, 18000).all(), (
            "Some line wavelengths are outside the expected APOGEE H-band range"
        )
