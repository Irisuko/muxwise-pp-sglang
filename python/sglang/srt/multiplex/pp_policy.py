"""FastPP-inspired, CPU-only policy primitives for the MuxWise PP lanes.

No chunking, pending-request suspension or transport changes live here. Times
are seconds. SM interaction features share observations across partitions;
otherwise partitions never visited by the static warmup could never become ready.
"""

from collections import deque

import numpy as np

from sglang.srt.multiplex.pp_chunk import lane_works


class OnlineRLS:
    def __init__(self, dim, bias_index, forgetting=0.9975, warmup=16, warm_start=True):
        self.theta = np.zeros(dim, dtype=np.float64)
        self.P = 1.25 * np.eye(dim)
        self.bias_index = bias_index
        self.forgetting = forgetting
        self.warmup = warmup
        self.warm_start = warm_start
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
        # Fit cold observations with ordinary RLS. Seeding a large intercept
        # and clipping all later residuals can collapse P before workload terms
        # are learned. A negative linear prediction then gets stuck at 1us:
        # clipping to 40% of that floor permits only a 0.4us correction forever.
        if not self.warm_start and not self.count:
            self.theta[self.bias_index] = observed
        raw = float(self.theta @ x)
        pred = max(raw, 1e-6)
        self.errors.append(abs(observed - pred) / max(observed, 1e-6))
        if self.warm_start and (self.count < self.warmup or raw <= 0):
            error = observed - raw
        else:
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


def choose_partition(
    sm_counts,
    decode_times,
    prefill_rates,
    required,
    pp_size,
    tpot,
    prefer_prefill=False,
):
    candidates = [i for i, (p, d) in enumerate(sm_counts) if p > 0 and d > 0]
    safe = [
        i
        for i in candidates
        if np.isfinite(decode_times[i]) and pp_size * decode_times[i] <= tpot
    ]
    # Meeting the arrival rate alone can leave requests spending seconds in a
    # local prefill FIFO. Use spare decode SLO headroom to drain that FIFO; the
    # exact same calibrated safety set still bounds the decode lane.
    if prefer_prefill and safe:
        return max(safe, key=lambda i: (sm_counts[i][0], -i))
    enough = [i for i in safe if np.isfinite(required) and prefill_rates[i] >= required]
    if enough:
        return min(enough, key=lambda i: (sm_counts[i][0], i))
    if safe:
        return max(safe, key=lambda i: (sm_counts[i][0], -i))
    return max(candidates, key=lambda i: (sm_counts[i][1], -i))


def work_bounded_inflight_limit(
    pp_size, hard_limit, token_budget, inflight_tokens, head_tokens
):
    """Extend the pipeline for small FIFO batches, not for more long prompts.

    The old 1.5*PP-depth operating point remains the soft limit. Extra slots
    require both existing work and the next FCFS request to fit a bounded token
    envelope. This is a scheduling gate only; allocator and future-output
    reservation checks still run unchanged for every admitted request.
    """
    soft = min(hard_limit, (3 * pp_size + 1) // 2)
    if (
        token_budget > 0
        and head_tokens <= token_budget
        and inflight_tokens <= soft * token_budget
    ):
        return hard_limit
    return soft


def decode_gap_layers(base_count, layer_seconds, gap_seconds):
    """Only shorten a slice if at least one whole layer fits the forecast gap.

    A layer is not preemptible. Fragmenting an already overlapping mixed-lane
    workload when even one layer cannot fit just adds scheduler/TP overhead.
    Keep 20% prediction margin and never exceed the original token-layer cap.
    """
    if (
        not np.isfinite(layer_seconds)
        or layer_seconds <= 0
        or not np.isfinite(gap_seconds)
        or gap_seconds <= 0
    ):
        return base_count
    fit = int(gap_seconds / (1.2 * layer_seconds))
    return min(base_count, fit) if fit >= 1 else base_count


def local_prefill_runnable(works):
    """The FIFO head owns local compute; remote/unmerged KV is not GPU work.

    Receiving counts as imminent compute so the partition is ready on arrival.
    Existing event/transfer fences still decide whether a switch is safe.
    """
    head = next((w for w in works if w.state != "done"), None)
    return head is not None and head.state in ("compute", "receiving")


def calibrated_decode_time(records, bs, mean_context):
    """Upper grid corner, never extrapolate from an unvisited SM partition.

    The table covers uniform and maximally ragged shapes at each mean context.
    Beyond graph/context coverage is unknown (infinity), not a tiny prediction.
    """
    eligible = [(b, c, t) for b, c, t in records if b >= bs and c >= mean_context]
    if not eligible:
        return float("inf")
    b = min(v[0] for v in eligible)
    c = min(v[1] for v in eligible)
    # A monotonic upper envelope avoids interpreting noisy profiling as a speedup
    # with increasing work. Sampling is outside the graph: reserve 1ms for it.
    return max(t for rb, rc, t in records if rb <= b and rc <= c)


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
        self.decode_calibration = {}
        self.decode_corrections = [deque(maxlen=128) for _ in scheduler.sm_counts]
        self.decode_scale = [1.2 for _ in scheduler.sm_counts]
        # One observation per complete local batch: a final slice includes
        # norm/logits/sampling and is not another interchangeable decoder layer.
        self.prefill = OnlineRLS(9, 4)
        self.prefill_slice = OnlineRLS(7, 3, warm_start=False)  # P2 diagnostic
        self.prefill_pending = {}
        self.prefill_legacy_batch_errors = deque(maxlen=128)
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
        if kind == "decode" and idx in self.decode_calibration:
            reference = calibrated_decode_time(
                self.decode_calibration[idx], base[0], base[1] * 1e4 / max(base[0], 1)
            )
            if np.isfinite(reference):
                self.decode_corrections[idx].append(seconds / reference)
                if self.samples_by_partition[idx][0] % 16 == 15:
                    self.decode_scale[idx] = max(
                        1.2, float(np.quantile(self.decode_corrections[idx], 0.9))
                    )
        self.samples_by_partition[idx][kind == "prefill"] += 1

    def observe_prefill_slice(self, key, base, idx, seconds, layers, final):
        old_base = [*base[:3], 1]
        old_features = self.features("prefill", old_base, idx)
        old_prediction = (
            self.prefill_slice.predict(old_features)
            if self.prefill_slice.count
            else seconds / layers
        )
        self.prefill_slice.update(old_features, seconds / layers)
        self.samples_by_partition[idx][1] += 1
        features = np.asarray(self.features("prefill", base, idx))
        accumulated = self.prefill_pending.setdefault(
            key, [np.zeros_like(features), 0.0, 0, 0.0]
        )
        accumulated[0] += features * layers
        accumulated[1] += seconds
        accumulated[2] += layers
        accumulated[3] += old_prediction * layers
        if final:
            x, elapsed, count, legacy = self.prefill_pending.pop(key)
            self.prefill_legacy_batch_errors.append(
                abs(elapsed - legacy) / max(elapsed, 1e-6)
            )
            self.prefill.update(x / count, elapsed / count)

    def _prefill_base(self):
        work = next(
            (w for w in lane_works(self.s._pp_prefills.values()) if w.state != "done"),
            None,
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
            len(lens),
            1,
        ]

    def partition(self, config, group_count, decode_bs, has_prefill):
        fallback = self.fallback.partition(config, group_count, decode_bs, has_prefill)
        if (
            not decode_bs
            or not has_prefill
            or not (
                self.decode_calibration or (self.decode.ready and self.prefill.ready)
            )
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
        if self.decode_calibration:
            workloads = [
                (len(mb.reqs), sum(req_context(r) for r in mb.reqs) / len(mb.reqs))
                for mb in s.running_mbs
                if mb.reqs
            ]
            td = [
                max(
                    calibrated_decode_time(
                        self.decode_calibration.get(i, []), n, context
                    )
                    for n, context in workloads
                )
                * self.decode_scale[i]
                + 0.001
                for i in range(group_count)
            ]
        else:
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
            s._pp_required if self.prefill.ready else float("inf"),
            s.pp_size,
            self.args.pdmux_pp_slo_tpot * self.args.pdmux_pp_tpot_coeff,
            prefer_prefill=getattr(self.args, "pdmux_pp_latency_partition", False),
        )
        if chosen != self.candidate:
            self.candidate, self.candidate_count = chosen, 0
        self.candidate_count += 1
        # Endpoint transitions cannot wait: prefill must have nonzero SMs.
        current_safe = (
            np.isfinite(td[s._pp_stream_idx])
            and s.pp_size * td[s._pp_stream_idx]
            <= self.args.pdmux_pp_slo_tpot * self.args.pdmux_pp_tpot_coeff
        )
        if 0 < s._pp_stream_idx < group_count - 1 and current_safe:
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
            base = [
                budget / 1e3,
                pb[1] * ratio**2,
                pb[2] * ratio,
                max(1, pb[3] * ratio),
                1,
            ]
            if (
                layers * self.prefill.predict(self.features("prefill", base, idx))
                <= target
            ):
                best = budget
        return best

    def decode_seconds(self, bs, ctx, idx):
        if self.decode_calibration:
            estimate = calibrated_decode_time(
                self.decode_calibration.get(idx, []), bs, ctx / max(bs, 1)
            )
            return estimate * self.decode_scale[idx] + 0.001
        if self.decode.ready:
            return self.decode.predict(self.features("decode", [bs, ctx / 1e4, 1], idx))
        return float("inf")

    def arrival_slice_layers(self, base_count, notice, now, host):
        if (
            not notice
            or notice["host"] != host
            or not notice["bs"]
            or now - notice["issued_at"] > self.args.pdmux_pp_slo_tpot
            or not self.prefill.ready
        ):
            return base_count
        layer = self.prefill.predict(
            self.features("prefill", self._prefill_base(), self.s._pp_stream_idx)
        )
        return decode_gap_layers(base_count, layer, notice["eta"] - now - 0.001)

    def merge_target(self, running_mbs):
        return self.fallback.merge_target(running_mbs)

    def slice_layers(self, *args):
        return self.fallback.slice_layers(*args)
