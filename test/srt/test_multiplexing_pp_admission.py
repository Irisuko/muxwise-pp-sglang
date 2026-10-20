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
