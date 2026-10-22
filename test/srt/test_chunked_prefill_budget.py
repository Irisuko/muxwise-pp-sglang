"""Regression for PP chunks resumed after the shared KV budget is exhausted."""
from types import SimpleNamespace as NS

import pytest

from sglang.srt.managers.schedule_policy import PrefillAdder


def make_adder(budget):
    return PrefillAdder(
        page_size=1,
        tree_cache=NS(evictable_size=lambda: 0),
        token_to_kv_pool_allocator=NS(available_size=lambda: 100),
        running_batch=NS(reqs=[NS(
            sampling_params=NS(max_new_tokens=100-budget), output_ids=[]
        )]),
        new_token_ratio=1.0,
        rem_input_tokens=16384,
        rem_chunk_tokens=2048,
    )


@pytest.mark.parametrize('budget', [-69, 0])
def test_defer_without_mutating_request_then_resume(budget):
    req = NS(extend_input_len=3000, prefix_indices=list(range(100)),
             fill_ids=list(range(3100)), sampling_params=NS(max_new_tokens=64))
    original = req.fill_ids.copy()
    adder = make_adder(budget)
    assert adder.add_chunked_req(req) is req
    assert adder.can_run_list == []
    assert req.extend_input_len == 3000 and req.fill_ids == original
    assert adder.log_input_tokens == 0
    # A decode completion makes room; the same request resumes its prefix.
    adder.token_to_kv_pool_allocator.available_size = lambda: 4096
    assert adder.add_chunked_req(req) is req
    assert adder.can_run_list == [req]
    assert req.extend_input_len == 2048
    assert req.fill_ids == original[:2148]
    assert adder.log_input_tokens == 2048


def test_deferred_chunk_releases_request_slot_once(monkeypatch):
    import sglang.srt.managers.scheduler as module
    monkeypatch.setattr(module, 'require_mlp_sync', lambda _: False)
    scheduler = module.Scheduler.__new__(module.Scheduler)

    class Chunk:
        req_pool_idx = 7

    scheduler.chunked_req = Chunk()
    cached, freed = [], []
    scheduler.tree_cache = NS(cache_unfinished_req=lambda req, **_: cached.append(req.req_pool_idx))
    scheduler.req_to_token_pool = NS(free=lambda slot: freed.append(slot))
    scheduler.tp_worker = NS(worker=NS(model_runner=NS(is_hybrid_gdn=False)))
    scheduler.last_batch = None
    scheduler.running_batch = NS(is_empty=lambda: True)
    scheduler.get_new_batch_prefill = lambda: None
    scheduler.server_args = NS()
    for _ in range(3):
        assert scheduler.get_next_batch_to_run() is None
    assert cached == [7] and freed == [7]
    assert scheduler.chunked_req.req_pool_idx is None
    # A subsequent actual chunk has a newly allocated slot and is released once.
    scheduler.chunked_req.req_pool_idx = 9
    scheduler.get_next_batch_to_run()
    scheduler.get_next_batch_to_run()
    assert cached == [7, 9] and freed == [7, 9]
