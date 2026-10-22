"""Deferred CUDA event samples; no synchronize in the scheduler's timing path."""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from sglang.srt.multiplex.pp_chunk import lane_works


class LaneTimings:
    def __init__(self, scheduler):
        self.s = scheduler
        self.pending = []
        trace = os.getenv("SGLANG_PDMUX_PP_TIMING_TRACE")
        self.trace = (
            open(f"{trace}.{scheduler.pp_rank}.jsonl", "w", buffering=1)
            if trace and scheduler.tp_rank == 0
            else None
        )
        rounds = os.getenv("SGLANG_PDMUX_PP_ROUND_TRACE")
        self.round_trace = (
            open(rounds, "w", buffering=1)
            if rounds and scheduler.pp_rank == 0 and scheduler.tp_rank == 0
            else None
        )
        request_trace = getattr(scheduler.server_args, "pdmux_pp_request_trace", None)
        self.request_trace = (
            open(f"{request_trace}.{scheduler.pp_rank}.jsonl", "w", buffering=1)
            if request_trace and scheduler.tp_rank == 0
            else None
        )
        self.round_previous = {}
        self.started = time.monotonic()
        self.phase = {"prefill": "empty", "decode": "control"}
        self.since = {lane: self.started for lane in self.phase}
        self.wall = {lane: defaultdict(float) for lane in self.phase}
        self.cuda = defaultdict(float)
        self.slices = defaultdict(lambda: [0, 0.0])
        self.partitions = defaultdict(float)
        self.partition_since = self.started
        self.partition = 0

    def request_event(self, event, work=None, rids=None, **fields):
        if self.request_trace is None or (rids is not None and not rids):
            return
        if work is not None:
            fields.update(
                batch=work.plan.batch_id,
                rids=work.plan.rids,
                batch_key=id(work.batch),
                state=work.state,
            )
            if getattr(work, "parent_id", None) is not None:
                fields.update(parent_batch=work.parent_id, chunk=work.chunk_index)
        elif rids is not None:
            fields["rids"] = rids
        self.request_trace.write(
            json.dumps(
                dict(
                    event=event,
                    at=time.time(),
                    mono=time.monotonic(),
                    stage=self.s.pp_rank,
                    **fields,
                )
            )
            + "\n"
        )

    def decode_launch(self, mb, batch):
        if self.round_trace is None:
            return
        now = time.monotonic()
        ids = tuple(r.rid for r in batch.reqs)
        previous = self.round_previous.get(mb)
        if previous and ids == previous[1]:
            self.round_trace.write(
                json.dumps(
                    dict(
                        at=time.time(),
                        mb=mb,
                        bs=len(ids),
                        ctx=previous[2],
                        partition=previous[3],
                        seconds=now - previous[0],
                    )
                )
                + "\n"
            )
        self.round_previous[mb] = (now, ids, batch.seq_lens_sum, self.s._pp_stream_idx)

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
            (w for w in lane_works(self.s._pp_prefills.values()) if w.state != "done"),
            None,
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
                if kind == "prefill":
                    self.request_event(
                        "cuda_slice",
                        batch_key=batch_key,
                        seconds=seconds,
                        layers=layers,
                        partition=idx,
                        detail=detail,
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
