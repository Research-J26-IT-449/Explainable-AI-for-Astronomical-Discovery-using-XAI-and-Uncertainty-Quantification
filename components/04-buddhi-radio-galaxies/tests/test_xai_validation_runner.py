"""Generated synthetic fixtures and metadata only; no research-image decoding."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from scripts import run_xai_validation_evaluation as run
from test_xai_evaluation_data import fixture  # Reuse the metadata-only archive fixture.


@pytest.fixture
def context(fixture):
    return run.VerifiedContext(fixture[0])


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        result = run._SyntheticModel().eval()
    for p in result.parameters():
        p.requires_grad_(False)
    return result


_PAYLOAD = None


@pytest.fixture
def payload(context, model):
    # One genuine end-to-end synthetic calculation is reused by storage tests.
    # Only checked metadata differs between their temporary archive fixtures.
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = run.evaluate_synthetic(context, 554, model)
    result = deepcopy(_PAYLOAD)
    result["provenance"] = run.primitive(context.authorize(554))
    return result


def test_real_module_integration_pairing_targets_and_four_gaps(payload):
    assert payload["origin"] == "generated_synthetic_fixture"
    assert payload["scientific_validation_evaluated"] is False
    target = payload["fixed_target_class"]
    assert target == payload["predicted_class"]
    assert len(payload["stability"]) == 30 and len(payload["sanity"]) == 31
    assert len(payload["emission"]) == 3
    for policy, result in payload["faithfulness"].items():
        assert result["status"] == "available", policy
        assert len(result["primary"]) == len(result["secondary"]) == 22
        for first, second in zip(result["primary"], result["secondary"]):
            assert first["curve"]["target_class"] == second["curve"]["target_class"] == target
            assert len(first["curve"]["deleted_logits"]) == 6
            assert len(second["curve"]["logits"]) == 21
            assert first["curve"]["deleted_logits"] == second["curve"]["logits"][1:7]
    record = run.restore_record(payload)
    metrics = dict(record.metrics)
    assert len(run.UNAVAILABLE) == len(set(run.UNAVAILABLE.values())) == 4
    for name, reason in run.UNAVAILABLE.items():
        assert metrics[name].status == "unavailable" and metrics[name].value is None
        assert metrics[name].reasons == (reason,)
    assert len(metrics[run.stats.RANDOM].repeats) == 20
    assert len(metrics["stability.noise.0.005.top0.1"].repeats) == 10
    assert len(metrics["sanity.stage6.jaccard"].repeats) == 5
    assert len(metrics["auc.black.probability.random"].repeats) == 20
    assert all(r["target_class"] == target for r in payload["stability"])
    assert all(r["fixed_target_class"] == target for r in payload["sanity"])
    assert sum(payload["stability_case_counts"].values()) == 30


def test_deterministic_serialization_no_tensor_or_nonfinite_outputs(payload):
    assert run.canonical(payload) == run.canonical(deepcopy(payload))
    assert json.loads(run.canonical(payload)) == payload
    with pytest.raises(ValueError):
        run.canonical({"pixels": np.zeros((150, 150))})
    with pytest.raises(ValueError):
        run.canonical({"value": float("nan")})


@pytest.mark.parametrize("index", [0, 673, 791, -1, 792, True])
def test_test_and_train_rejected_before_image_or_model(context, monkeypatch, index):
    image = Mock(side_effect=AssertionError("image generated before authorization"))
    forward = Mock(side_effect=AssertionError("CAM before authorization"))
    monkeypatch.setattr(run, "_synthetic_image", image)
    monkeypatch.setattr(run.gradcam, "generate_gradcam", forward)
    with pytest.raises(PermissionError):
        run.evaluate_synthetic(context, index, object())
    image.assert_not_called()
    forward.assert_not_called()


def test_locked_test_selection_precedes_model_construction(fixture, monkeypatch):
    build = Mock(side_effect=AssertionError("mock constructed for TEST"))
    monkeypatch.setattr(run, "_SyntheticModel", build)
    with pytest.raises(PermissionError, match="Locked TEST"):
        run.run_synthetic(fixture[0], (673,))
    build.assert_not_called()


def test_dry_run_exact_allowlist_no_images_models_or_writes(fixture, monkeypatch):
    root, _, splits, _ = fixture
    import pickle
    from PIL import Image
    blocked = Mock(side_effect=AssertionError("pixels/model access in metadata dry-run"))
    for module, name in ((pickle, "load"), (pickle, "loads"), (Image, "open"), (torch, "load"),
                         (run, "_SyntheticModel"), (run, "_synthetic_image")):
        monkeypatch.setattr(module, name, blocked)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run.dry_run(root)
    assert result["validation_indices"] == splits["validation"]
    assert len(result["validation_indices"]) == 119
    assert result["validation_class_counts"] == {"0": 53, "1": 66}
    assert result["engineering_count"] == 40
    assert result["scientific_validation_coverage"] == {"eligible": 119, "evaluated": 0}
    assert result["image_pixels_decoded"] == result["model_forwards"] == result["locked_test_images_accessed"] == 0
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    blocked.assert_not_called()


@pytest.mark.parametrize("key", ["checkpoint", "split", "archive"])
def test_provenance_hash_mismatches_rejected(fixture, key):
    root, config, _, _ = fixture
    (root / config["provenance"][key]).write_bytes(b"changed frozen bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        run.VerifiedContext(root)


def test_protocol_hash_and_module_drift_rejected(fixture, monkeypatch):
    root, config, _, _ = fixture
    monkeypatch.setattr(run.faith, "FRACTIONS", (.1,)*6)
    with pytest.raises(ValueError, match="protocol/API"):
        run.VerifiedContext(root)
    config["statistics"]["bootstrap"]["seed"] = 99
    (root / run.data.CONFIG_PATH).write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        run.VerifiedContext(root)


def test_after_load_drift_fails_before_synthetic_generation(fixture, context, model, monkeypatch):
    root, config, _, _ = fixture
    (root / config["provenance"]["split"]).write_bytes(b"changed")
    image = Mock(side_effect=AssertionError("must not generate"))
    monkeypatch.setattr(run, "_synthetic_image", image)
    with pytest.raises(ValueError, match="hash mismatch"):
        run.evaluate_synthetic(context, 554, model)
    image.assert_not_called()


def test_incorrect_allowlist_or_label_fails_closed(fixture, monkeypatch):
    original = run.data.load_evaluation_data
    access = original(fixture[0])
    entries = access.validation.entries
    for altered in (entries[:-1], (replace(entries[0], ground_truth_label=2), *entries[1:])):
        invalid = replace(access, validation=replace(access.validation, entries=altered))
        monkeypatch.setattr(run.data, "load_evaluation_data", lambda root: invalid)
        with pytest.raises(ValueError, match="allowlist/labels"):
            run.VerifiedContext(fixture[0])


def test_mock_only_no_arbitrary_research_model(context):
    with pytest.raises(PermissionError, match="mock model"):
        run.evaluate_synthetic(context, 554, torch.nn.Linear(2, 2).eval())


def test_atomic_publication_duplicates_and_resume(context, payload):
    store = run.ResultStore(context, (554,))
    store.write(554, payload)
    path = store.directory / store._name(554)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        store.write(554, payload)
    assert path.read_bytes() == before
    assert not list(store.directory.glob("*.pending"))
    resumed = run.ResultStore(context, (554,), resume=True)
    assert resumed.read(554) == payload
    result = resumed.finish()
    assert result == resumed.finish()
    assert result["scientific_validation_coverage"]["evaluated"] == 0
    for name, reason in run.UNAVAILABLE.items():
        summary = result["metric_summaries"][name]["overall"]
        assert summary["available"] == 0 and summary["unavailable"] == 1
        assert summary["unavailable_reasons"] == [[reason, 1]]


@pytest.mark.parametrize("change", ["digest", "payload", "run", "partial_json"])
def test_resume_corruption_rejected(context, payload, change):
    store = run.ResultStore(context, (554,))
    store.write(554, payload)
    path = store.directory / store._name(554)
    envelope = json.loads(path.read_bytes())
    if change == "partial_json":
        path.write_bytes(b'{"payload":')
    else:
        if change == "digest":
            envelope["payload_sha256"] = "0"*64
        elif change == "payload":
            envelope["payload"]["predicted_class"] = 1-envelope["payload"]["predicted_class"]
        else:
            envelope["run_sha256"] = "0"*64
        path.write_bytes(run.canonical(envelope))
    with pytest.raises(ValueError):
        run.ResultStore(context, (554,), resume=True)


@pytest.mark.parametrize("change", ["stability", "sanity_duplicate", "full_curve", "primary", "missing_gap", "target"])
def test_partial_or_inconsistent_module_results_rejected(context, payload, change):
    store = run.ResultStore(context, (554,))
    broken = deepcopy(payload)
    if change == "stability":
        broken["stability"].pop()
    elif change == "sanity_duplicate":
        broken["sanity"][1] = broken["sanity"][2]
    elif change == "full_curve":
        broken["faithfulness"]["black"]["secondary"][0]["curve"]["logits"].pop()
    elif change == "primary":
        broken["faithfulness"]["black"]["primary"][0]["aopc"]["logit"] += 1
    elif change == "missing_gap":
        broken["record"]["metrics"] = [r for r in broken["record"]["metrics"] if r[0] != "sanity_random_map"]
    else:
        broken["fixed_target_class"] = 1-broken["fixed_target_class"]
    with pytest.raises(ValueError):
        store.write(554, broken)
    assert not (store.directory / store._name(554)).exists()


def test_atomic_failure_no_incomplete_final_or_protected_writes(context, payload, monkeypatch):
    before = run.protected_hashes(context.root)
    store = run.ResultStore(context, (554,))
    monkeypatch.setattr(run.os, "link", Mock(side_effect=OSError("simulated publication interruption")))
    with pytest.raises(OSError):
        store.write(554, payload)
    assert not (store.directory / store._name(554)).exists()
    assert not list(store.directory.glob("*.pending"))
    assert run.protected_hashes(context.root) == before
    with pytest.raises(PermissionError):
        store._publish("../../models/model.pth", {})


def test_stale_partial_files_and_missing_completion_rejected(context, payload):
    store = run.ResultStore(context, (554,))
    with pytest.raises(FileNotFoundError):
        store.finish()
    (store.directory / "crashed.pending").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Partial"):
        run.ResultStore(context, (554,), resume=True)


def test_duplicate_request_and_unapproved_run_modes(context):
    with pytest.raises(ValueError, match="Duplicate"):
        run.ResultStore(context, (554, 554))
    with pytest.raises(SystemExit):
        run.main(["--mode", "real"])
    with pytest.raises(SystemExit):
        run.main(["--mode", "dry-run", "--index", "554"])


def test_run_resume_does_not_repeat_completed_work(fixture, context, payload, monkeypatch):
    root = fixture[0]
    store = run.ResultStore(context, (554,))
    store.write(554, payload)
    evaluate = Mock(side_effect=AssertionError("completed image reevaluated"))
    monkeypatch.setattr(run, "evaluate_synthetic", evaluate)
    result = run.run_synthetic(root, (554,), resume=True)
    assert result["summary"]["synthetic_images_completed"] == 1
    evaluate.assert_not_called()


def test_model_and_inputs_preserved_and_repeated_synthetic_execution(context, model, payload):
    before = run._state(model)
    rng = torch.get_rng_state().clone()
    repeat = run.evaluate_synthetic(context, 554, model)
    assert run._same_state(before, run._state(model))
    assert torch.equal(rng, torch.get_rng_state())
    assert repeat == payload


def test_degenerate_reference_keeps_independent_controls_and_diagnostics(context, model, monkeypatch):
    original = run.gradcam.generate_gradcam
    def degenerate(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, status="degenerate", raw_cam=torch.zeros_like(result.raw_cam),
                       upsampled_heatmap=torch.zeros_like(result.upsampled_heatmap), heatmap=torch.zeros_like(result.heatmap))
    monkeypatch.setattr(run.gradcam, "generate_gradcam", degenerate)
    monkeypatch.setattr(run.sanity, "generate_gradcam", degenerate)
    result = run.evaluate_synthetic(context, 554, model)
    record = run.restore_record(result)
    assert dict(record.metrics)[run.stats.CAM].value is None
    assert dict(record.metrics)[run.stats.RANDOM].status == "available"
    assert dict(record.metrics)["sanity.stage1.jaccard"].status == "unavailable"
    for value in result["faithfulness"].values():
        assert value["status"] == "unavailable" and len(value["independent_controls"]) == 21
    assert all(r["status"] == "unavailable" for r in result["stability"])
    assert all(r["original_class_probability_change"] is not None for r in result["stability"])
    run.ResultStore(context, (554,)).write(554, result)


def test_unexpected_failure_propagates_no_partial_record(context, model, monkeypatch):
    before = run._state(model)
    fail = Mock(side_effect=RuntimeError("unexpected synthetic API failure"))
    monkeypatch.setattr(run.faith, "evaluate_full_rankings", fail)
    with pytest.raises(RuntimeError, match="unexpected synthetic"):
        run.evaluate_synthetic(context, 554, model)
    assert run._same_state(before, run._state(model))


def test_workload_estimate_is_planning_only():
    cost = run.workload()
    assert cost["primary_deletion_forwards_uncached"] == 15708
    assert cost["full_three_replacement_deletion_forwards"] == 157080
    assert cost["total_cam_calls_with_current_controls"] == 7497
    assert cost["primary_reuses_full_first_six_points"]


def test_no_protected_output_directory_allowed(context, payload):
    store = run.ResultStore(context, (554,))
    store.directory = context.root / "outputs/models"
    before = run.protected_hashes(context.root)
    with pytest.raises(PermissionError):
        store.write(554, payload)
    assert run.protected_hashes(context.root) == before


@pytest.mark.parametrize("change", ["scalar", "classification", "transformed_class"])
def test_partial_scalars_or_bad_classes_are_not_published(context, payload, change):
    broken = deepcopy(payload)
    if change == "scalar":
        broken["record"]["metrics"] = [r for r in broken["record"]["metrics"] if r[0] != run.stats.CAM]
    elif change == "classification":
        broken["classification_correct"] = not broken["classification_correct"]
    else:
        broken["stability"][0]["transformed_predicted_class"] = 2
    store = run.ResultStore(context, (554,))
    with pytest.raises(ValueError):
        store.write(554, broken)


def test_completed_summary_drift_and_run_metadata_drift_rejected(context, payload):
    store = run.ResultStore(context, (554,))
    store.write(554, payload)
    store.finish()
    (store.directory / "summary.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="summary integrity"):
        store.finish()
    (store.directory / "run.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="provenance/schema"):
        run.ResultStore(context, (554,), resume=True)


def test_stage_counts_and_identity_metrics_remain_in_summaries(payload):
    result = run.summarize_results([payload])
    assert result["complete_frozen_protocol_coverage"] is False
    assert result["stability_case_counts"] == payload["stability_case_counts"]
    assert "sanity.identity.jaccard" in result["metric_summaries"]
    assert "emission.primary.area" in result["metric_summaries"]
    for name in ("sanity.identity.jaccard", run.stats.CAM):
        assert result["metric_summaries"][name] == run.primitive(run.stats.summarize((run.restore_record(payload),), name))


class ResourceClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def resource_monitor(**kwargs):
    return run.PilotResources(memory_reader=lambda: {"working_set_bytes": 100, "process_peak_bytes": 120}, **kwargs)


def test_resource_defaults_are_cooperative_pilot_thresholds():
    monitor = resource_monitor()
    assert monitor.max_bytes == 2*1024**3 and monitor.max_seconds == 900
    report = monitor.report()
    assert set(run.RESOURCE_FAMILIES) <= set(report["families"])
    assert "cannot be interrupted" in report["enforcement"]
    assert "before/after family only" in report["sanity_checkpoints"]
    assert report["process_lifetime_peak_bytes"] == 120


@pytest.mark.parametrize("snapshot", [{}, {"status": "unavailable"}, {"working_set_bytes": None},
                                      {"working_set_bytes": float("nan")}, {"working_set_bytes": -1},
                                      {"working_set_bytes": True}, None])
def test_unreliable_memory_fails_closed(snapshot):
    with pytest.raises(run.ResourceSafetyError, match="measurement unavailable"):
        run.PilotResources(memory_reader=lambda: snapshot)


@pytest.mark.parametrize("kwargs", [{"max_working_set_bytes": 0}, {"max_working_set_bytes": True},
                                    {"max_image_seconds": 0}, {"max_image_seconds": float("inf")},
                                    {"max_image_seconds": float("nan")}, {"max_image_seconds": True}])
def test_invalid_resource_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        resource_monitor(**kwargs)


def test_working_set_violation_stops_before_operation():
    snapshot = {"working_set_bytes": 100}
    monitor = run.PilotResources(memory_reader=lambda: snapshot, max_working_set_bytes=200)
    snapshot["working_set_bytes"] = 201
    operation = Mock()
    with pytest.raises(run.ResourceSafetyError, match="working-set limit"):
        monitor.call("stability_robustness", operation)
    operation.assert_not_called()


def test_cooperative_timeout_detects_after_blocking_call_not_hard_interrupt():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)
    completed = []

    def blocking_fixture():
        clock.now = 901  # Simulate elapsed time inside an uninterruptible call.
        completed.append(True)
        return "finished"

    with pytest.raises(run.ResourceSafetyError, match="cooperative checkpoint"):
        monitor.call("parameter_randomization", blocking_fixture)
    assert completed == [True]  # No hard interruption is claimed.
    assert monitor.families["parameter_randomization"]["failures"] == 1


def test_scoring_loop_checks_each_callback_and_stops_next_iteration():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)
    seen = []

    def score(value):
        seen.append(value)
        clock.now += 500
        return value

    guarded = monitor.scorer(score)
    assert guarded(1) == 1
    with pytest.raises(run.ResourceSafetyError):
        guarded(2)
    with pytest.raises(run.ResourceSafetyError):
        guarded(3)
    assert seen == [1, 2]


def test_family_timing_exact_and_no_telemetry_in_function_result():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)
    result = {"deterministic": 7}

    def calculate():
        clock.now += 2.5
        return result

    assert monitor.call("radio_emission_alignment", calculate) is result
    assert result == {"deterministic": 7}
    assert monitor.report()["families"]["radio_emission_alignment"] == {
        "elapsed_seconds": 2.5, "inclusive_seconds": 2.5, "calls": 1, "failures": 0}


def test_full_synthetic_integration_has_same_records_with_resource_telemetry(context, model, payload):
    monitor = resource_monitor()
    before = run._state(model)
    result = run._evaluate_image(context, 554, model, run._synthetic_image(554), context.authorize(554), resources=monitor)
    assert run.canonical(result) == run.canonical(payload)
    assert run._same_state(before, run._state(model))
    report = monitor.report()
    for name in run.RESOURCE_FAMILIES:
        if name != "frozen_ood_diagnostics":
            assert report["families"][name]["calls"] > 0, name
    assert report["families"]["frozen_ood_diagnostics"]["calls"] == 0
    assert "telemetry" not in result and result["approved_unavailable"] == run.UNAVAILABLE
    assert report["memory_samples"] > 2*3*22*20


def test_ood_timer_observes_actual_diagnostic_only_and_restores_profiler(model):
    from src.ood.ood_detector import OODDetector, OODThresholds
    detector = OODDetector(model)
    detector.feature_mean = torch.zeros(512)
    detector.feature_variance = torch.ones(512)
    detector.thresholds = OODThresholds(1e8, 1e8, 0.99)
    monitor = resource_monitor()
    tensor = run.faith.normalize_intensity(run._synthetic_image(554))
    before = run._state(model)
    previous = sys.getprofile()
    expected = detector.validate_tensor(tensor)
    observed = monitor.ood_scorer(detector.validate_tensor)(tensor)
    assert observed == expected
    assert sys.getprofile() is previous
    assert monitor.report()["families"]["frozen_ood_diagnostics"]["calls"] == 1
    assert run._same_state(before, run._state(model))
    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        monitor.ood_scorer(Mock(side_effect=RuntimeError("synthetic scoring failure")))(tensor)
    assert sys.getprofile() is previous


def test_existing_profiler_is_not_overwritten():
    monitor = resource_monitor()
    previous = sys.getprofile()
    observer = lambda *_: None
    try:
        sys.setprofile(observer)
        with pytest.raises(run.ResourceSafetyError, match="profiler conflicts"):
            monitor.ood_scorer(Mock())(None)
        assert sys.getprofile() is observer
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("failure", [run.ResourceSafetyError("resource limit"), ValueError("non-finite critical output"),
                                    RuntimeError("unexpected model-state drift")])
def test_first_image_failure_stops_second_no_partial_publication(context, model, monkeypatch, failure):
    from src.xai import execution_adapter
    image_access = Mock(side_effect=lambda i: run._synthetic_image(i))

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            self.indices, self.profile = (554, 555), run.SYNTHETIC_PROFILE

        def model(self):
            return model

        image = staticmethod(image_access)

        def diagnostic_scorer(self, source):
            return run.faith.ModelScorer(source), []

    monkeypatch.setattr(execution_adapter, "RealExecutionAdapter", FakeAdapter)
    monkeypatch.setattr(run, "VerifiedContext", lambda _: context)
    monkeypatch.setattr(run, "PilotResources", lambda **_: resource_monitor_original())
    monkeypatch.setattr(run, "_evaluate_image", Mock(side_effect=failure))
    before = run.protected_hashes(context.root)
    with pytest.raises(type(failure)) as caught:
        run.run_real(context.root, (554, 555), max_images=2, approval={"approved": True})
    assert [call.args[0] for call in image_access.call_args_list] == [554]
    assert not list((context.root/"outputs/xai").rglob("image_*.json"))
    assert not list((context.root/"outputs/xai").rglob("summary.json"))
    assert caught.value.pilot_telemetry["completed_images"] == []
    assert run.protected_hashes(context.root) == before


# Save the class for tests that replace the runner's factory, avoiding recursion.
_PilotResources = run.PilotResources


def resource_monitor_original():
    return _PilotResources(memory_reader=lambda: {"working_set_bytes": 100, "process_peak_bytes": 120})


def test_timeout_in_record_validation_prevents_atomic_image_publication(context, payload, monkeypatch):
    store = run.ResultStore(context, (554,))
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)
    validate = store._validate

    def slow_validate(*args):
        validate(*args)
        clock.now = 901

    monkeypatch.setattr(store, "_validate", slow_validate)
    with pytest.raises(run.ResourceSafetyError):
        store.write(554, payload, resources=monitor)
    assert not (store.directory/store._name(554)).exists()


def test_statistics_timeout_does_not_publish_summary(context, payload, monkeypatch):
    store = run.ResultStore(context, (554,))
    store.write(554, payload)
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)

    def slow_summary(*args, **kwargs):
        clock.now = 901
        return {"not_published": True}

    monkeypatch.setattr(run, "summarize_results", slow_summary)
    with pytest.raises(run.ResourceSafetyError):
        store.finish(resources=monitor)
    assert not (store.directory/"summary.json").exists()
    assert store.read(554) == payload
    with pytest.raises(FileExistsError):
        store.write(554, payload)


def test_same_family_nested_timing_counts_four_seconds_once():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)

    def inner():
        clock.now += 2

    def outer():
        clock.now += 1
        monitor.call("full_deletion_curves_auc", inner)
        clock.now += 1

    monitor.call("full_deletion_curves_auc", outer)
    report = monitor.report()
    assert report["elapsed_seconds"] == 4
    assert report["instrumented_exclusive_seconds"] == 4
    assert report["families"]["full_deletion_curves_auc"] == {
        "elapsed_seconds": 4, "inclusive_seconds": 4, "calls": 2, "failures": 0}


def test_different_family_nested_timing_has_additive_exclusive_totals():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)

    def primary():
        clock.now += 2

    def full():
        clock.now += 1
        monitor.call("primary_aopc", primary)
        clock.now += 1

    monitor.call("full_deletion_curves_auc", full)
    report = monitor.report()
    assert report["families"]["full_deletion_curves_auc"]["elapsed_seconds"] == 2
    assert report["families"]["full_deletion_curves_auc"]["inclusive_seconds"] == 4
    assert report["families"]["primary_aopc"]["elapsed_seconds"] == 2
    assert report["instrumented_exclusive_seconds"] == report["elapsed_seconds"] == 4


def test_degenerate_cam_independent_controls_accounting_regression(monkeypatch):
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock, max_image_seconds=10000)
    image = run._synthetic_image(554)
    rankings = (run.faith.rank_brightness(image, 554), run.faith.random_rankings(554))
    original_auc = run.faith.deletion_auc

    def auc(curve):
        clock.now += 2
        return original_auc(curve)

    def score(tensor):
        clock.now += 1
        return torch.tensor([[2.0, 1.0]])

    monkeypatch.setattr(run.faith, "deletion_auc", auc)
    result = monitor.call("full_deletion_curves_auc", run._independent_controls,
                          image, rankings, score, "border_median", (2.0, 1.0), 0, monitor)
    assert len(result) == 21
    assert clock.now == 21*20+21*2
    report = monitor.report()
    assert report["families"]["full_deletion_curves_auc"]["inclusive_seconds"] == clock.now
    assert report["families"]["full_deletion_curves_auc"]["elapsed_seconds"] == clock.now
    assert report["instrumented_exclusive_seconds"] == clock.now


@pytest.mark.parametrize("destination", ["image", "summary"])
def test_last_resource_check_after_fsync_prevents_publication(context, payload, monkeypatch, destination):
    store = run.ResultStore(context, (554,))
    if destination == "summary":
        store.write(554, payload)
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)
    original_fsync = run.os.fsync
    publication = Mock(wraps=run.os.link)

    def expire_after_fsync(fd):
        original_fsync(fd)
        clock.now = 901

    monkeypatch.setattr(run.os, "fsync", expire_after_fsync)
    monkeypatch.setattr(run.os, "link", publication)
    with pytest.raises(run.ResourceSafetyError):
        if destination == "image":
            store.write(554, payload, resources=monitor)
        else:
            store.finish(resources=monitor)
    publication.assert_not_called()
    name = store._name(554) if destination == "image" else "summary.json"
    assert not (store.directory/name).exists()
    assert not list(store.directory.glob("*.pending"))


def test_detected_memory_violation_stays_latched_before_publication(context, payload):
    snapshot = {"working_set_bytes": 100}
    monitor = run.PilotResources(memory_reader=lambda: snapshot, max_working_set_bytes=200)
    store = run.ResultStore(context, (554,))
    snapshot["working_set_bytes"] = 201
    with pytest.raises(run.ResourceSafetyError):
        monitor.check()
    snapshot["working_set_bytes"] = 100
    with pytest.raises(run.ResourceSafetyError, match="remains stopped"):
        store.write(554, payload, resources=monitor)
    assert not (store.directory/store._name(554)).exists()
    assert monitor.report()["failure"]["reason_code"] == "memory_limit_exceeded"


def test_failure_diagnostics_capture_family_elapsed_and_memory():
    clock = ResourceClock()
    monitor = resource_monitor(clock=clock)

    def expire():
        clock.now = 901

    with pytest.raises(run.ResourceSafetyError) as caught:
        monitor.call("parameter_randomization", expire)
    diagnostic = run._failure_diagnostics(caught.value, monitor)
    assert diagnostic["reason_code"] == "time_limit_exceeded"
    report = diagnostic["resource_report"]
    assert report["elapsed_seconds"] == 901
    assert report["failure"]["metric_family"] == "parameter_randomization"
    assert report["last_memory_measurement"]["working_set_bytes"] == 100
    assert report["last_memory_measurement"]["sample_elapsed_seconds"] == 901


@pytest.mark.parametrize("kind", ["resource", "execution"])
def test_cli_failure_emits_sanitized_operational_json_only(monkeypatch, capsys, kind):
    secret = "private-path/secret-token"
    error = ValueError(secret) if kind == "execution" else run.ResourceSafetyError(secret)
    monitor = resource_monitor()
    error.pilot_telemetry = {"completed_images": [], "failure": run._failure_diagnostics(error, monitor)}
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"{}")  # No receipt file exists or is created.
    monkeypatch.setattr(run, "run_real", Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        run.main(["--mode", "real", "--index", "31", "--index", "50", "--max-images", "2",
                  "--approval-file", "synthetic-no-file.json"])
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert output.out == "" and secret not in output.err and "Traceback" not in output.err
    report = json.loads(output.err)
    assert report["status"] == "failed" and report["scientific_completion_reported"] is False
    assert report["telemetry"]["completed_images"] == []
    assert report["telemetry"]["failure"]["resource_report"]["last_memory_measurement"]["status"] == "available"


def test_cli_missing_receipt_has_sanitized_setup_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=OSError("sensitive private path")))
    with pytest.raises(SystemExit):
        run.main(["--mode", "real", "--index", "31", "--max-images", "1", "--approval-file", "absent.json"])
    output = capsys.readouterr()
    assert output.out == "" and "sensitive" not in output.err
    diagnostic = json.loads(output.err)["telemetry"]
    assert diagnostic["execution_elapsed_seconds"] >= 0
    assert diagnostic["failure"]["resource_report"]["last_memory_measurement"]["status"] == "not_sampled"


def test_setup_resource_failure_attaches_diagnostics_without_context_or_pixels(monkeypatch):
    blocked = Mock(side_effect=AssertionError("setup continued despite unavailable memory"))
    monkeypatch.setattr(run, "VerifiedContext", blocked)
    monkeypatch.setattr(run, "PilotResources", lambda **_: _PilotResources(memory_reader=lambda: None))
    with pytest.raises(run.ResourceSafetyError) as caught:
        run.run_real(indices=(31, 50), max_images=2, approval={"approved": True})
    blocked.assert_not_called()
    report = caught.value.pilot_telemetry
    assert report["execution_elapsed_seconds"] >= 0
    assert report["failure"]["reason_code"] == "memory_measurement_unavailable"
    assert report["failure"]["resource_report"]["last_memory_measurement"]["status"] == "unavailable"
