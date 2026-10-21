"""Explicit migration availability boundaries; never fetch private research artifacts.

Skipped checks remain blocked, not passed. Synthetic assertions are unchanged.
"""
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def pytest_collection_modifyitems(items):
    for item in items:
        filename=Path(str(item.fspath)).name
        reason=None
        if filename in {'test_dataset.py','test_preprocessing.py'}:
            reason='BLOCKED: real MiraBest archive is intentionally not distributed; no download or real image access during migration.'
        elif filename=='test_pseudocolor.py' and item.originalname=='test_known_train_samples_retain_fr_predictions':
            reason='BLOCKED: requires private real TRAIN images, saved split and historical checkpoint; migration is synthetic-only.'
        elif filename=='test_public_demo.py' and item.originalname=='test_private_gallery_references_existing_precomputed_figures_only':
            reason='BLOCKED: individual radio overlays excluded pending redistribution clearance; aggregate UI remains tested.'
        blocked_artifact_tests={
            ('test_auto_radio.py','test_requested_demo_regression_exact_model_tensors'),
            ('test_auto_radio.py','test_generated_calibration_has_train_validation_provenance_only'),
            ('test_auto_radio.py','test_original_ood_artifact_unchanged_during_auto_prediction'),
            ('test_demo.py','test_selected_checkpoint_loads_in_evaluation_mode'),
            ('test_external_radio.py','test_exported_mirabest_png_actual_classifier_and_ood_consistency'),
            ('test_ood_detector.py','test_locked_checkpoint_and_final_metrics_exist_and_are_not_ood_outputs'),
            ('test_rotation_augmentation_experiment.py','test_experiment_never_uses_test_and_preserves_final_artifacts'),
        }
        if (filename,item.originalname) in blocked_artifact_tests:
            reason='BLOCKED: excluded real image exports, historical checkpoints/calibration or private final artifacts required; no substitutes or downloads.'
        if reason:item.add_marker(pytest.mark.skip(reason=reason))
