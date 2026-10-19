"""Deferred CUDA event samples; no synchronize in the scheduler's timing path."""

import torch


class LaneTimings:
    def __init__(self, scheduler):
        self.s = scheduler
        self.pending = []

    def begin(self, kind, base, layers=1):
        lane = 1 if kind == "decode" else 0
        # Endpoint streams are ordinary full-device streams. A newly merged
        # decode can execute before a safe partition switch; do not label that
        # transition observation as a zero-SM lane in the regression model.
        if self.s.sm_counts[self.s._pp_stream_idx][lane] == 0:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return kind, base, layers, self.s._pp_stream_idx, start

    def end(self, sample):
        if sample is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pending.append((*sample, end))

    def drain(self):
        pending, self.pending = self.pending, []
        for sample in pending:
            kind, base, layers, idx, start, end = sample
            if end.query():
                self.s._pp_models.observe(
                    kind, base, idx, start.elapsed_time(end) / (1000 * layers)
                )
            else:
                self.pending.append(sample)
