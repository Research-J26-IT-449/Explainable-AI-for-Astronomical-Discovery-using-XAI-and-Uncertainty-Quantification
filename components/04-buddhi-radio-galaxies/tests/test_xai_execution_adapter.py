"""Synthetic serialization, checkpoints and models ONLY; no research pixels."""

from copy import deepcopy
import io
import json
from pathlib import Path
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts import run_xai_validation_evaluation as run
from src.xai import execution_adapter as adapter
from test_xai_evaluation_data import fixture, archive_bytes
from test_xai_validation_runner import payload, model  # Existing genuine synthetic integration fixture.


def row_member(start=528, *, data_override=None):
    labels = [0 if i < 245 or 554 <= i < 607 or 673 <= i < 726 else 1 for i in range(start, start+88)]
    filenames = [f"BestCat_proc/Cropped/{100 if c == 0 else 200}_{i}.123_+041.815.png"
                 for i, c in zip(range(start, start+88), labels)]
    rows = [np.full((150, 150), i % 256, np.uint8) for i in range(start, start+88)]
    return pickle.dumps({"batch_label": "synthetic", "labels": labels,
                         "data": rows if data_override is None else data_override,
                         "filenames": filenames}, protocol=3).replace(b"numpy._core", b"numpy.core")


@pytest.fixture
def context(fixture):
    return run.VerifiedContext(fixture[0])


@pytest.fixture
def row_context(fixture):
    root, config, _, pin = fixture
    raw = archive_bytes({name: row_member(i*88) for i, name in enumerate(run.data.MEMBERS)})
    (root/config["provenance"]["archive"]).write_bytes(raw)
    config["provenance"].update(archive_sha256=run.data.digest(raw), archive_md5=run.data.digest(raw, "md5"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        net = adapter.resnet_cnn.ResNetCNN()
    checkpoint = root/config["provenance"]["checkpoint"]
    torch.save({"model_state_dict": net.state_dict()}, checkpoint)
    config["provenance"]["checkpoint_sha256"] = run.data.digest(checkpoint.read_bytes())
    pin()
    return run.VerifiedContext(root)


def approve(context, indices=(554,), bound=1):
    return {**adapter.approval_request(context, indices, bound), "approved": True}


class GuardedStream(io.BytesIO):
    """Raise on ANY read touching image payloads, including unselected rows."""
    def __init__(self, raw, ranges, selected=None):
        super().__init__(raw)
        self.ranges, self.selected, self.image_reads = ranges, selected, []

    def read(self, n=-1):
        begin, end = self.tell(), self.tell()+n if n >= 0 else len(self.getbuffer())
        for i, span in enumerate(self.ranges):
            if begin < span.offset+span.length and end > span.offset:
                assert i == self.selected, f"Unauthorized row {i} was read"
                assert begin == span.offset and n == 22500
                self.image_reads.append(i)
        return super().read(n)


def test_metadata_parser_never_reads_pixels_or_executes_pickle(monkeypatch):
    raw = row_member()
    layout = adapter.inspect_member_layout(io.BytesIO(raw), len(raw))
    assert len(layout.rows) == 88
    blocked = Mock(side_effect=AssertionError("Executable/bulk decoder called"))
    monkeypatch.setattr(pickle, "load", blocked)
    monkeypatch.setattr(pickle, "loads", blocked)
    monkeypatch.setattr(np, "frombuffer", blocked)
    stream = GuardedStream(raw, layout.rows)
    assert adapter.inspect_member_layout(stream, len(raw)) == layout
    assert stream.image_reads == []
    blocked.assert_not_called()


def test_authorized_row_decodes_only_one_separate_payload():
    raw = row_member()
    layout = adapter.inspect_member_layout(io.BytesIO(raw), len(raw))
    stream = GuardedStream(raw, layout.rows, selected=26)
    image = adapter.decode_authorized_row(stream, len(raw), 26, expected_label=0,
                                         expected_filename=layout.filenames[26])
    assert image.shape == (150, 150) and image.dtype == np.uint8
    assert np.all(image == 554 % 256)
    assert stream.image_reads == [26]


@pytest.mark.parametrize("value", [np.zeros((88, 150, 150), np.uint8), b"bulk pixels",
                                  [np.zeros((150, 150), np.float32)]*88,
                                  [np.zeros((149, 150), np.uint8)]*88,
                                  [np.zeros((150, 150), np.uint8)]*88])
def test_bulk_wrong_dtype_shape_and_aliased_arrays_rejected(value):
    raw = row_member(data_override=value)
    with pytest.raises(ValueError):
        adapter.inspect_member_layout(io.BytesIO(raw), len(raw))


@pytest.mark.parametrize("raw", [b"\x80\x03cos\nsystem\n.", b"\x80\x04}", b"\x80\x03}",
                                 b"\x80\x03h\xff.", b"\x80\x03B\xff\xff\xff\xff"])
def test_malicious_or_malformed_unsupported_encodings_fail_closed(raw):
    with pytest.raises(ValueError):
        adapter.inspect_member_layout(io.BytesIO(raw), len(raw))


def test_metadata_mismatch_before_selected_pixel_read():
    raw = row_member()
    layout = adapter.inspect_member_layout(io.BytesIO(raw), len(raw))
    stream = GuardedStream(raw, layout.rows)
    with pytest.raises(ValueError, match="metadata/provenance"):
        adapter.decode_authorized_row(stream, len(raw), 26, expected_label=1, expected_filename="bad")
    assert not stream.image_reads


def test_approval_gate_default_disabled_before_context_or_decoding(monkeypatch):
    blocked = Mock(side_effect=AssertionError("Access before approval"))
    monkeypatch.setattr(run, "VerifiedContext", blocked)
    with pytest.raises(PermissionError, match="disabled"):
        run.run_real(indices=(554,), max_images=1)
    with pytest.raises(PermissionError, match="disabled"):
        adapter.RealExecutionAdapter(object(), (554,), max_images=1)
    blocked.assert_not_called()


@pytest.mark.parametrize("bound,indices", [(None, (554,)), (0, (554,)), (120, (554,)), (True, (554,)),
                                          (1, (554, 555)), (1, ()), (2, (554, 554)), (2, (555, 554))])
def test_mandatory_image_bounds_and_explicit_selection(context, bound, indices):
    with pytest.raises(ValueError):
        adapter.approval_request(context, indices, bound)


@pytest.mark.parametrize("index", [0, 673, 791, -1, 792, True])
def test_unauthorized_indices_rejected_before_row_or_model_access(context, monkeypatch, index):
    blocked = Mock(side_effect=AssertionError("Decoded before allowlist"))
    monkeypatch.setattr(adapter, "decode_authorized_row", blocked)
    with pytest.raises((PermissionError, ValueError)):
        adapter.approval_request(context, (index,), 1)
    blocked.assert_not_called()


def test_gate_bound_to_checkpoint_split_archive_protocol_and_preprocessing(context):
    receipt = approve(context)
    assert adapter.validate_approval(context, (554,), 1, receipt)["indices"] == [554]
    for key in ("context_sha256", "preprocessing", "indices", "max_images"):
        bad = deepcopy(receipt)
        bad[key] = "changed"
        with pytest.raises(ValueError, match="mismatch"):
            adapter.validate_approval(context, (554,), 1, bad)


def test_unsupported_metadata_only_archive_has_no_bulk_fallback(context, monkeypatch):
    blocked = Mock(side_effect=AssertionError("unsafe fallback"))
    monkeypatch.setattr(pickle, "load", blocked)
    with pytest.raises(ValueError, match="protocol-3"):
        adapter.RealExecutionAdapter(context, (554,), max_images=1, approval=approve(context))
    blocked.assert_not_called()


def test_authorized_adapter_has_no_construction_decode_then_exact_preprocessing(row_context, monkeypatch):
    before = run.protected_hashes(row_context.root)
    blocked = Mock(side_effect=AssertionError("constructor materialized pixels"))
    original = adapter.decode_authorized_row
    monkeypatch.setattr(adapter, "decode_authorized_row", blocked)
    real = adapter.RealExecutionAdapter(row_context, (554,), max_images=1, approval=approve(row_context))
    blocked.assert_not_called()
    for index in (673, 555, 0):
        with pytest.raises(PermissionError):
            real.image(index)
    blocked.assert_not_called()
    monkeypatch.setattr(adapter, "decode_authorized_row", original)
    intensity = real.image(554)
    assert intensity.dtype == np.float32 and np.all(intensity == np.float32(42)/255)
    expected = adapter.preprocess.build_transforms()[1](np.full((150, 150), 42, np.uint8)).unsqueeze(0)
    assert torch.equal(run.faith.normalize_intensity(intensity), expected)
    assert run.protected_hashes(row_context.root) == before


def test_private_safe_checkpoint_model_preservation_rng_and_no_cache(row_context):
    before = run.protected_hashes(row_context.root)
    real = adapter.RealExecutionAdapter(row_context, (554,), max_images=1, approval=approve(row_context))
    rng = torch.random.get_rng_state().clone()
    model = real.model()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert real.model() is model and all(not m.training for m in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    state = run._state(model)
    cam = run.gradcam.generate_gradcam(model, run.faith.normalize_intensity(real.image(554)))
    assert cam.predicted_class in (0, 1)
    assert run._same_state(state, run._state(model))
    assert run.protected_hashes(row_context.root) == before


@pytest.mark.parametrize("key", ["checkpoint", "archive", "split"])
def test_after_approval_provenance_drift_rejected_before_decode(row_context, monkeypatch, key):
    receipt = approve(row_context)
    config = __import__("yaml").safe_load((row_context.root/run.data.CONFIG_PATH).read_bytes())
    (row_context.root/config["provenance"][key]).write_bytes(b"corrupt synthetic fixture")
    blocked = Mock(side_effect=AssertionError("decode on mismatch"))
    monkeypatch.setattr(adapter, "decode_authorized_row", blocked)
    with pytest.raises(ValueError):
        adapter.RealExecutionAdapter(row_context, (554,), max_images=1, approval=receipt)
    blocked.assert_not_called()


def test_real_store_namespace_and_resume_integrity(context, payload):
    profile = {"mode": "real", "origin": "authorized_validation_row", "notice": adapter.REAL_NOTICE,
               "namespace": "real_validation", "scientific_validation_evaluated": True,
               "execution": adapter.approval_request(context, (554,), 1)}
    result = deepcopy(payload)
    result.update(origin=profile["origin"], notice=profile["notice"], scientific_validation_evaluated=True,
                  ood_diagnostics={"status": "available", "classifier_curves_filtered": False,
                                   "calls": [{"accepted": False}]})
    before = run.protected_hashes(context.root)
    store = run.ResultStore(context, (554,), profile=profile)
    assert "real_validation" in str(store.directory)
    store.write(554, result)
    assert store.read(554) == result
    with pytest.raises(FileExistsError):
        store.write(554, result)
    resumed = run.ResultStore(context, (554,), profile=profile, resume=True)
    assert resumed.read(554) == result
    summary = resumed.finish()
    assert summary["scientific_validation_coverage"] == {"eligible": 119, "evaluated": 1}
    for name, reason in run.UNAVAILABLE.items():
        assert summary["approved_unavailable"][name] == reason
    path = store.directory/store._name(554)
    path.write_text("corrupt")
    with pytest.raises(ValueError):
        run.ResultStore(context, (554,), profile=profile, resume=True)
    assert run.protected_hashes(context.root) == before


def test_shared_orchestration_targets_pairing_and_all_modules(context, model, monkeypatch):
    image = run._synthetic_image(554)
    before = run._state(model)
    diagnostics = []
    score = run.faith.ModelScorer(model)

    def diagnostic_score(tensor):
        diagnostics.append({"accepted": False})
        return score(tensor)

    profile = {"mode": "real", "origin": "synthetic_adapter_test", "notice": "synthetic fixture ONLY",
               "scientific_validation_evaluated": False}
    result = run._evaluate_image(context, 554, model, image, context.authorize(554), profile=profile,
                                 scorer=diagnostic_score, diagnostics=diagnostics)
    assert len(diagnostics) == 3*(1+22*20)
    assert result["ood_diagnostics"]["classifier_curves_filtered"] is False
    assert len(result["sanity"]) == 31 and len(result["stability"]) == 30
    assert all(r["fixed_target_class"] == result["predicted_class"] for r in result["sanity"])
    assert result["approved_unavailable"] == run.UNAVAILABLE
    assert run._same_state(before, run._state(model))


def test_telemetry_is_separate_from_result_and_serializable():
    value, telemetry = adapter.instrument(lambda: {"scientific": 7})
    assert value == {"scientific": 7}
    assert telemetry["elapsed_seconds"] >= 0 and telemetry["cpu_seconds"] >= 0
    assert json.loads(run.canonical(telemetry)) == telemetry


def test_real_cli_cannot_accidentally_execute_without_approval(monkeypatch):
    blocked = Mock(side_effect=AssertionError("real path entered"))
    monkeypatch.setattr(run, "run_real", blocked)
    with pytest.raises(SystemExit):
        run.main(["--mode", "real", "--index", "554", "--max-images", "1"])
    blocked.assert_not_called()
