"""Deferred CUDA event samples; no synchronize in the scheduler's timing path."""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class LaneTimings:
    def __init__(self, scheduler):
        self.s = scheduler
        self.pending = []
        trace = os.getenv("SGLANG_PDMUX_PP_TIMING_TRACE")
        self.trace = (
            open(f"{trace}.{scheduler.pp_rank}.jsonl", "w", buffering=1)
            if trace
            else None
        )
        self.started = time.monotonic()
        self.phase = {"prefill": "empty", "decode": "control"}
        self.since = {lane: self.started for lane in self.phase}
        self.wall = {lane: defaultdict(float) for lane in self.phase}
        self.cuda = defaultdict(float)
        self.slices = defaultdict(lambda: [0, 0.0])
        self.partitions = defaultdict(float)
        self.partition_since = self.started
        self.partition = 0

    def mark(self, lane, phase):
        now = time.monotonic()
        self.wall[lane][self.phase[lane]] += now - self.since[lane]
        self.phase[lane], self.since[lane] = phase, now

    @contextmanager
    def region(self, lane, phase):
        previous = self.phase[lane]
        self.mark(lane, phase)
        try:
            yield
        finally:
            self.mark(lane, previous)

    def prefill_state(self):
        work = next(
            (w for w in self.s._pp_prefills.values() if w.state != "done"), None
        )
        phase = (
            "empty"
            if work is None
            else {
                "waiting": "upstream",
                "receiving": "upstream",
                "offering": "ready",
                "sending": "send",
                "compute": "compute_slot",
            }[work.state]
        )
        self.mark("prefill", phase)

    def switch(self, idx):
        now = time.monotonic()
        self.partitions[self.partition] += now - self.partition_since
        self.partition, self.partition_since = idx, now

    def snapshot(self):
        for lane, phase in self.phase.items():
            self.mark(lane, phase)
        self.switch(self.partition)
        return dict(
            elapsed=time.monotonic() - self.started,
            wall={k: dict(v) for k, v in self.wall.items()},
            cuda=dict(self.cuda),
            slices={k: v.copy() for k, v in self.slices.items()},
            partitions=dict(self.partitions),
        )

    def begin(self, kind, base, layers=1, detail="all", batch_key=None):
        lane = 1 if kind == "decode" else 0
        # Endpoint streams are ordinary full-device streams. A newly merged
        # decode can execute before a safe partition switch; do not label that
        # transition observation as a zero-SM lane in the regression model.
        idx = self.s._pp_stream_idx
        if self.s.sm_counts[idx][lane] == 0:
            idx = None  # Count compute time, but exclude endpoint transitions from RLS.
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return kind, base, layers, idx, detail, batch_key, start

    def end(self, sample):
        if sample is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pending.append((*sample, end))

    def drain(self):
        pending, self.pending = self.pending, []
        for sample in pending:
            kind, base, layers, idx, detail, batch_key, start, end = sample
            if end.query():
                seconds = start.elapsed_time(end) / 1000
                if self.trace and kind == "prefill":
                    self.trace.write(
                        json.dumps(
                            dict(
                                at=time.time(),
                                base=base,
                                layers=layers,
                                partition=idx,
                                detail=detail,
                                seconds=seconds,
                            )
                        )
                        + "\n"
                    )
                self.cuda[kind] += seconds
                key = f"{kind}_{detail}"
                self.slices[key][0] += 1
                self.slices[key][1] += seconds / layers
                if kind == "prefill" and idx is not None:
                    self.s._pp_models.observe_prefill_slice(
                        batch_key, base, idx, seconds, layers, detail == "final"
                    )
                elif kind == "decode" and idx is not None:
                    self.s._pp_models.observe(kind, base, idx, seconds / layers)
            else:
                self.pending.append(sample)
