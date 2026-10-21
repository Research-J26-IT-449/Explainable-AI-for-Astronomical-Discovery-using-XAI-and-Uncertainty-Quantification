"""Operational epoch logging around the unchanged, single-process trainer.

Durations cover training plus validation, excluding scheduler/checkpoint/plot IO.
Rows flush at the next epoch or successful return, after scheduler updates.
This temporary binding is not suitable for concurrent trainers in one process.
"""

import json
from pathlib import Path
from time import perf_counter

from src.models import train


class EpochTelemetry:
    def __init__(self, path, *, clock=perf_counter):
        self.path = Path(path)
        self.clock = clock
        self.pending = None
        self.epoch = 0

    def __enter__(self):
        self.stream = self.path.open("x", encoding="utf-8")
        self.original = train._run_epoch
        train._run_epoch = self._run
        return self

    def _flush(self):
        if self.pending is not None:
            self.pending["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            line = json.dumps(self.pending, sort_keys=True, allow_nan=False)
            self.stream.write(line + "\n")
            self.stream.flush()
            print(line, flush=True)
            self.pending = None

    def _run(self, model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            self._flush()
            self.epoch += 1
            self.started = self.clock()
            self.optimizer = optimizer
        result = self.original(model, loader, criterion, device, optimizer)
        if optimizer is not None:
            self.training = result
        else:
            self.pending = dict(epoch=self.epoch, train_loss=self.training[0],
                                train_accuracy=self.training[1], validation_loss=result[0],
                                validation_accuracy=result[1],
                                epoch_duration_seconds=self.clock() - self.started)
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        train._run_epoch = self.original
        try:
            if exc_type is None:
                self._flush()
        finally:
            self.stream.close()
