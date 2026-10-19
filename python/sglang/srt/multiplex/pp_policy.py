"""FastPP-inspired, CPU-only policy primitives for the MuxWise PP lanes.

No chunking, pending-request suspension or transport changes live here. Times
are seconds. SM interaction features share observations across partitions;
otherwise partitions never visited by the static warmup could never become ready.
"""

from collections import deque

import numpy as np


class OnlineRLS:
    def __init__(self, dim, bias_index, forgetting=0.9975, warmup=16):
        self.theta = np.zeros(dim, dtype=np.float64)
        self.P = 1.25 * np.eye(dim)
        self.bias_index = bias_index
        self.forgetting = forgetting
        self.warmup = warmup
        self.count = 0
        self.errors = deque(maxlen=128)

    @property
    def ready(self):
        return self.count >= self.warmup

    @property
    def mape(self):
        return float(np.mean(self.errors)) if self.errors else None

    def predict(self, x):
        return max(float(self.theta @ np.asarray(x)), 1e-6)

    def update(self, x, observed):
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)) or not np.isfinite(observed) or observed <= 0:
            return
        # No startup profiler: seed only the intercept from the first real
        # observation. A zero prediction with relative clipping cannot learn.
        if not self.count:
            self.theta[self.bias_index] = observed
        pred = self.predict(x)
        self.errors.append(abs(observed - pred) / max(observed, 1e-6))
        error = np.clip(observed - pred, -0.4 * abs(pred), 0.4 * abs(pred))
        px = self.P @ x
        gain = px / (self.forgetting + x @ px)
        self.theta += gain * error
        self.P = (self.P - np.outer(gain, x @ self.P)) / self.forgetting
        self.P = (self.P + self.P.T) * 0.5
        self.count += 1


def sm_features(base, sm, total_sm):
    """Base features end in 1; append workload * (total_sm / lane_sm - 1)."""
    scale = total_sm / max(sm, 1) - 1
    return [*base, *(v * scale for v in base[:-1])]


def choose_partition(sm_counts, decode_times, prefill_rates, required, pp_size, tpot):
    candidates = [i for i, (p, d) in enumerate(sm_counts) if p > 0 and d > 0]
    safe = [i for i in candidates if pp_size * decode_times[i] <= tpot]
    enough = [i for i in safe if prefill_rates[i] >= required]
    if enough:
        return min(enough, key=lambda i: (sm_counts[i][0], i))
    if safe:
        return max(safe, key=lambda i: (sm_counts[i][0], -i))
    return max(candidates, key=lambda i: (sm_counts[i][1], -i))


def required_throughput(waiting, arrivals, now, ttft, tpot, fresh=0.25, window=10):
    """waiting/arrivals contain (arrival_time, effective_input_tokens)."""
    stale = [(now - t, n) for t, n in waiting if now - t >= fresh * ttft]
    tokens = sum(n for _, n in stale)
    budget = ttft - np.mean([t for t, _ in stale]) if stale else ttft
    catch_up = (float("inf") if budget <= tpot else tokens / budget) if tokens else 0
    return catch_up + sum(n for t, n in arrivals if t >= now - window) / window


def req_context(req):
    return len(req.origin_input_ids) + len(req.output_ids)


def req_load(req, a, b):
    return a + b * req_context(req)


def balanced_assignment(reqs, batches, pending, a, b, max_bs):
    loads = [sum(req_load(r, a, b) for r in rs if not r.finished()) for rs in batches]
    counts = [sum(not r.finished() for r in rs) for rs in batches]
    for i, rs in enumerate(pending):
        loads[i] += sum(req_load(r, a, b) for r in rs if not r.finished())
        counts[i] += sum(not r.finished() for r in rs)
    targets = []
    for req in reqs:
        eligible = [i for i, n in enumerate(counts) if n < max_bs]
        assert eligible, "Prefill admission exceeded total microbatch capacity"
        target = min(eligible, key=lambda i: (loads[i], i))
        targets.append(target)
        if not req.finished():
            counts[target] += 1
            loads[target] += req_load(req, a, b)
    return targets


def rebalance_pair(batches, a, b, unit):
    loads = [sum(req_load(r, a, b) for r in rs if not r.finished()) for rs in batches]
    source = max(range(len(loads)), key=lambda i: (loads[i], -i))
    target = min(range(len(loads)), key=lambda i: (loads[i], i))
    return (source, target) if loads[source] - loads[target] > 1.4 * unit * a else None


def migration_rids(source, target, a, b, unit, max_bs):
    """Enter above 1.4 units, stop in the 0.4-unit band or without improvement."""
    source = [r for r in source if not r.finished()]
    target = [r for r in target if not r.finished()]
    delta = sum(req_load(r, a, b) for r in source) - sum(
        req_load(r, a, b) for r in target
    )
    if delta <= 1.4 * unit * a:
        return []
    moved = []
    while source and len(target) + len(moved) < max_bs and delta > 0.4 * unit * a:
        req = min(source, key=lambda r: (abs(delta - 2 * req_load(r, a, b)), r.rid))
        new_delta = delta - 2 * req_load(req, a, b)
        if abs(new_delta) >= abs(delta):
            break
        source.remove(req)
        moved.append(req.rid)
        delta = new_delta
    return moved


class ALPMuxPPPolicy:
    def __init__(self, scheduler, fallback):
        self.s = scheduler
        self.args = scheduler.server_args
        self.fallback = fallback
        self.decode = OnlineRLS(5, 2)
        self.prefill = OnlineRLS(7, 3)
        self.samples_by_partition = [[0, 0] for _ in scheduler.sm_counts]
        self.last_switch = 0
        self.candidate = None
        self.candidate_count = 0

    def features(self, kind, base, idx):
        counts = self.s.sm_counts[idx]
        return sm_features(base, counts[1 if kind == "decode" else 0], sum(counts))

    def observe(self, kind, base, idx, seconds):
        model = self.decode if kind == "decode" else self.prefill
        model.update(self.features(kind, base, idx), seconds)
        self.samples_by_partition[idx][kind == "prefill"] += 1

    def _prefill_base(self):
        work = next(
            (w for w in self.s._pp_prefills.values() if w.state != "done"), None
        )
        if work:
            lens, past = work.batch.extend_lens, work.batch.prefix_lens
        else:
            reqs = self.s.waiting_queue[:16]
            lens = [
                getattr(r, "_pp_effective_tokens", len(r.origin_input_ids))
                for r in reqs
            ]
            past = [max(0, len(r.origin_input_ids) - n) for r, n in zip(reqs, lens)]
        if not lens:
            lens, past = [self.args.pdmux_pp_min_budget], [0]
        return [
            sum(lens) / 1e3,
            sum(n * n for n in lens) / 1e6,
            sum(n * p for n, p in zip(lens, past)) / 1e6,
            1,
        ]

    def partition(self, config, group_count, decode_bs, has_prefill):
        fallback = self.fallback.partition(config, group_count, decode_bs, has_prefill)
        if (
            not decode_bs
            or not has_prefill
            or not (self.decode.ready and self.prefill.ready)
        ):
            return fallback
        s = self.s
        bs, ctx = max(
            (
                (len(mb.reqs), sum(req_context(r) for r in mb.reqs))
                for mb in s.running_mbs
            ),
            key=lambda v: v[0] + v[1] / 1e4,
        )
        db = [bs, ctx / 1e4, 1]
        pb = self._prefill_base()
        layers = (
            s.tp_worker.model_runner.model.end_layer
            - s.tp_worker.model_runner.model.start_layer
        )
        td = [
            self.decode.predict(self.features("decode", db, i))
            for i in range(group_count)
        ]
        rates = [
            pb[0]
            * 1e3
            / (layers * self.prefill.predict(self.features("prefill", pb, i)))
            for i in range(group_count)
        ]
        chosen = choose_partition(
            s.sm_counts,
            td,
            rates,
            s._pp_required,
            s.pp_size,
            self.args.pdmux_pp_slo_tpot * self.args.pdmux_pp_tpot_coeff,
        )
        if chosen != self.candidate:
            self.candidate, self.candidate_count = chosen, 0
        self.candidate_count += 1
        # Endpoint transitions cannot wait: prefill must have nonzero SMs.
        if 0 < s._pp_stream_idx < group_count - 1:
            if (
                s._pp_decode_steps - self.last_switch
                < self.args.pdmux_pp_min_residency_steps
                or self.candidate_count < self.args.pdmux_pp_hysteresis_steps
            ):
                return s._pp_stream_idx
        return chosen

    def admission_budget(self, scheduler):
        args = self.args
        upper = min(args.pdmux_pp_max_budget, scheduler.max_prefill_tokens)
        if not args.pdmux_pp_alp_admission or not self.prefill.ready:
            return min(scheduler.max_prefill_tokens, args.pdmux_pp_max_budget)
        lower = min(args.pdmux_pp_min_budget, upper)
        pb = self._prefill_base()
        n = max(pb[0] * 1e3, 1)
        pressure = min(1, scheduler._pp_waiting_tokens / max(upper, 1))
        target = args.pdmux_pp_slo_ttft / (2 * scheduler.pp_size) * (1 + pressure)
        layers = (
            scheduler.tp_worker.model_runner.model.end_layer
            - scheduler.tp_worker.model_runner.model.start_layer
        )
        idx = scheduler._pp_stream_idx
        if scheduler.sm_counts[idx][0] == 0:
            idx = 1
        best = lower
        # A bounded grid avoids assuming a noisy fitted polynomial is monotonic.
        for budget in list(range(lower, upper, 256)) + [upper]:
            ratio = budget / n
            base = [budget / 1e3, pb[1] * ratio**2, pb[2] * ratio, 1]
            if (
                layers * self.prefill.predict(self.features("prefill", base, idx))
                <= target
            ):
                best = budget
        return best

    def merge_target(self, running_mbs):
        return self.fallback.merge_target(running_mbs)

    def slice_layers(self, *args):
        return self.fallback.slice_layers(*args)
