"""Allocator-based admission accounting. Counts decisions, never sums contexts as KV."""

import time
from collections import Counter, defaultdict


class AdmissionStats:
    def __init__(self, scheduler):
        self.s = scheduler
        self.decisions = Counter()
        self.blocked = Counter()
        self.stops = Counter()
        self.wall = defaultdict(float)
        self.phase = "empty"
        self.since = time.monotonic()
        self.next_sample = 0.0
        self.series = []
        self.last = {}

    @staticmethod
    def capacity_reason(available, needed, reserve):
        # Match the scheduler's strict safety inequality exactly.
        if needed >= available:
            return "kv"
        if reserve + needed >= available:
            return "decode_reserve"
        return None

    def record(self, reason, admitted=0, **details):
        now = time.monotonic()
        self.wall[self.phase] += now - self.since
        self.phase = "admitted" if admitted else reason
        self.since = now
        self.decisions[self.phase] += 1
        if reason not in ("empty", "admitted"):
            self.stops[reason] += 1
            if not admitted:
                self.blocked[reason] += 1
        self.last = dict(reason=reason, admitted=admitted, **details)
        if now >= self.next_sample:
            self.series.append(dict(at=time.time(), **self.allocator(), **self.last))
            self.next_sample = now + 0.5

    def allocator(self):
        s = self.s
        free = s.token_to_kv_pool_allocator.available_size()
        evictable = s.tree_cache.evictable_size()
        return dict(
            capacity=s.max_total_num_tokens,
            used=s.max_total_num_tokens - free,
            free=free,
            evictable=evictable,
            protected=s.max_total_num_tokens - free - evictable,
            waiting=len(s.waiting_queue),
            inflight=len(s._pp_prefills),
            inflight_new_tokens=sum(
                w.batch.extend_num_tokens for w in s._pp_prefills.values()
            ),
            unfinished_prefill_tokens=sum(
                w.batch.extend_num_tokens - getattr(w, "completed_tokens", 0)
                for w in s._pp_prefills.values()
            ),
            free_reqs=s.req_to_token_pool.available_size(),
        )

    def snapshot(self):
        now = time.monotonic()
        self.wall[self.phase] += now - self.since
        self.since = now
        samples, self.series = self.series, []
        return dict(
            decisions=dict(self.decisions),
            blocked=dict(self.blocked),
            stops=dict(self.stops),
            wall=dict(self.wall),
            allocator=self.allocator(),
            last=self.last.copy(),
            series=samples,
        )
