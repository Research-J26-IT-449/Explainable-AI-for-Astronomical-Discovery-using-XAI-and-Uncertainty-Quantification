"""Synthetic operational logging checks: no datasets or model execution."""
import json
from types import SimpleNamespace

import pytest

from src.models import train
from src.models.training_telemetry import EpochTelemetry


def test_metrics_calls_duration_and_post_scheduler_lr(tmp_path, monkeypatch):
    calls = []
    def original(*args):
        calls.append(args)
        return (0.4, 0.75) if args[-1] is not None else (0.5, 0.6)
    monkeypatch.setattr(train, "_run_epoch", original)
    ticks = iter([10, 16, 20, 29])
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0003}])
    path = tmp_path / "epochs.jsonl"
    with EpochTelemetry(path, clock=lambda: next(ticks)):
        for _ in range(2):
            assert train._run_epoch(None, None, None, None, optimizer) == (0.4, 0.75)
            assert train._run_epoch(None, None, None, None) == (0.5, 0.6)
            optimizer.param_groups[0]["lr"] /= 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(calls) == 4
    assert [r["epoch"] for r in rows] == [1, 2]
    assert [r["epoch_duration_seconds"] for r in rows] == [6, 9]
    assert [r["learning_rate"] for r in rows] == [0.00015, 0.000075]
    assert all(r["train_loss"] == 0.4 and r["validation_accuracy"] == 0.6 for r in rows)
    assert train._run_epoch is original
    with pytest.raises(FileExistsError):
        with EpochTelemetry(path):
            pass


def test_failure_restores_binding_without_successful_epoch(tmp_path, monkeypatch):
    def original(*args):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(train, "_run_epoch", original)
    path = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with EpochTelemetry(path):
            train._run_epoch(None, None, None, None, SimpleNamespace(param_groups=[]))
    assert train._run_epoch is original
    assert path.read_bytes() == b""
