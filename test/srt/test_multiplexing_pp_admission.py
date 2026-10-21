"""CPU checks for mutually exclusive capacity attribution and real pool occupancy."""

from types import SimpleNamespace

from sglang.srt.multiplex.pp_admission import AdmissionStats


def test_capacity_attribution():
    assert AdmissionStats.capacity_reason(100, 100, 0) == "kv"
    assert AdmissionStats.capacity_reason(100, 70, 30) == "decode_reserve"
    assert AdmissionStats.capacity_reason(100, 70, 29) is None


def test_real_allocator_and_decisions():
    s = SimpleNamespace(
        max_total_num_tokens=100,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 10),
        tree_cache=SimpleNamespace(evictable_size=lambda: 60),
        req_to_token_pool=SimpleNamespace(available_size=lambda: 5),
        waiting_queue=[object()],
        _pp_prefills={0: SimpleNamespace(batch=SimpleNamespace(extend_num_tokens=7))},
    )
    stats = AdmissionStats(s)
    stats.record("kv", needed=80)
    stats.record("budget", admitted=1)
    stats.record("empty")
    snapshot = stats.snapshot()
    assert snapshot["blocked"] == {"kv": 1}
    assert snapshot["stops"] == {"kv": 1, "budget": 1}
    assert snapshot["allocator"]["used"] == 90
    assert snapshot["allocator"]["protected"] == 30
    assert snapshot["allocator"]["inflight_new_tokens"] == 7
    assert snapshot["series"] and not stats.snapshot()["series"]


def test_extra_slot_rechecks_stale_prefix_estimate_and_replays_failed_probe():
    from types import SimpleNamespace as NS
    from sglang.srt.multiplex.multiplexing_pp import SchedulerMultiplexPPMixin
    from sglang.srt.multiplex.pp_admission import AdmissionStats

    s = SchedulerMultiplexPPMixin()
    s.pp_size = 2
    s._pp_max_prefills = 4
    s._pp_admission_budget = 8
    s._pp_batch_id = 17
    s.max_running_requests = 128
    s.server_args = NS(
        pdmux_pp_adaptive_inflight=True,
        pdmux_pp_reserve_decode=True,
        pdmux_pp_balance_merge=True,
        max_micro_batch_size=None,
    )
    r = NS(
        rid="long",
        origin_input_ids=list(range(9)),
        fill_ids=list(range(9)),
        prefix_indices=[],
        last_node="node",
        extend_input_len=9,
        _pp_effective_tokens=1,
        sampling_params=NS(max_new_tokens=1),
        init_next_round_input=lambda _: None,
    )
    s.waiting_queue = [r]
    s.running_mbs = [NS(reqs=[]), NS(reqs=[])]
    s._pp_pending_merges = [[], []]
    s._pp_prefills = {i: NS(batch=NS(reqs=[], extend_num_tokens=4)) for i in range(3)}
    s.mux_pp_policy = NS(merge_target=lambda _: 0)
    s.req_to_token_pool = NS(available_size=lambda: 128)
    s.token_to_kv_pool_allocator = NS(available_size=lambda: 10000)
    locks = []
    s.tree_cache = NS(
        inc_lock_ref=lambda n: locks.append(("lock", n)),
        dec_lock_ref=lambda n: locks.append(("unlock", n)),
        evictable_size=lambda: 0,
    )
    recorded = []
    s._pp_admission_stats = NS(
        capacity_reason=AdmissionStats.capacity_reason,
        record=lambda *a, **kw: recorded.append((a, kw)),
    )
    s._pp_build_prefill = lambda *_: (_ for _ in ()).throw(
        AssertionError("must not allocate")
    )
    plan = s._pp_make_plan()
    assert plan.rids == [] and plan.probes == [("long", 9, 0)]
    assert recorded[-1][0] == ("inflight", 0)
    assert locks == [("lock", "node"), ("unlock", "node")]
    assert s._pp_batch_id == 17 and s.waiting_queue == [r]
