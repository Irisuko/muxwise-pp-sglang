"""Chunk coverage, shared KV ownership and terminal-only sampling regressions."""

import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.multiplex.multiplexing_pp import (
    PrefillPlan,
    PrefillWork,
    SchedulerMultiplexPPMixin,
)
from sglang.srt.multiplex.pp_chunk import chunk_batch, lane_works, plan_chunks


@pytest.mark.parametrize(
    "lengths,budget", [([31], 4), ([1, 3, 9, 2], 4), ([5, 2], 100), ([2, 3], 1)]
)
def test_exact_ordered_coverage(lengths, budget):
    chunks = plan_chunks(lengths, budget)
    seen = []
    finals = []
    for chunk in chunks:
        assert 0 < chunk.tokens <= budget
        assert len({i for i, _, _ in chunk.spans}) == len(chunk.spans)
        for i, off, n in chunk.spans:
            seen += [(i, j) for j in range(off, off + n)]
            if off + n == lengths[i]:
                finals.append(i)
        terminal_rows = [
            j for j, (i, off, n) in enumerate(chunk.spans) if off + n == lengths[i]
        ]
        assert terminal_rows == list(range(len(terminal_rows)))
    assert seen == [(i, j) for i, n in enumerate(lengths) for j in range(n)]
    assert finals == list(range(len(lengths)))


def test_views_preserve_full_allocation_and_request_state():
    class Batch(NS):
        pass

    reqs = [NS(rid=str(i), to_abort=False) for i in range(3)]
    parent = Batch(
        reqs=reqs,
        extend_lens=[3, 9, 2],
        extend_num_tokens=14,
        prefix_lens=[2, 7, 0],
        input_ids=torch.arange(14),
        out_cache_loc=torch.arange(40, 54),
        req_pool_indices=torch.tensor([5, 2, 9]),
        orig_seq_lens=torch.tensor([5, 16, 2]),
        multimodal_inputs=[None] * 3,
        model_config=NS(vocab_size=100),
        device="cpu",
    )
    owners = []

    def sampling(batch, _):
        owners.append(weakref.ref(batch))
        return [r.rid for r in batch.reqs]

    with patch(
        "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
        side_effect=sampling,
    ):
        children = [chunk_batch(parent, c) for c in plan_chunks(parent.extend_lens, 4)]
    assert [b.prefix_lens for b in children] == [[2, 7], [8], [12], [0]]
    assert [b.seq_lens.tolist() for b in children] == [[5, 8], [12], [16], [2]]
    assert [b.sampling_info for b in children] == [["0"], None, ["1"], ["2"]]
    assert [b.pdmux_pp_append_chunk for b in children] == [False, True, True, False]
    assert all(owner() is not None for owner in owners)
    assert torch.equal(
        torch.cat([b.out_cache_loc for b in children]), parent.out_cache_loc
    )
    assert all(
        b.out_cache_loc.untyped_storage().data_ptr()
        == parent.out_cache_loc.untyped_storage().data_ptr()
        for b in children
    )
    assert parent.extend_lens == [3, 9, 2] and parent.prefix_lens == [2, 7, 0]
    reqs[1].to_abort = True
    assert all(r.to_abort for b in children for r in b.reqs if r.rid == "1")


def test_final_chunk_retirement_and_lane_fifo():
    s = SchedulerMultiplexPPMixin()
    parent = PrefillWork(
        PrefillPlan(8, [], ["a"]), NS(split_prefill_finished=False), "compute"
    )
    parent.chunks = [
        PrefillWork(
            PrefillPlan(i, [], ["a"]), NS(), "done", event=i, parent_id=8, chunk_index=i
        )
        for i in range(3)
    ]
    s._pp_prefills = {8: parent}
    assert list(lane_works(s._pp_prefills.values())) == parent.chunks
    s._pp_finish_chunk(parent.chunks[0])
    assert not parent.batch.split_prefill_finished
    s._pp_finish_chunk(parent.chunks[-1])
    assert (
        parent.batch.split_prefill_finished
        and parent.event == 2
        and parent.state == "done"
    )


def test_partial_chunk_skips_norm_and_lm_head():
    from test_multiplexing_pp import TestSplitPrefill

    model = TestSplitPrefill().model(0, 8)

    def forbidden(*args):
        raise AssertionError("intermediate chunk must not evaluate logits")

    model.model.norm = forbidden
    model.logits_processor = forbidden
    batch = NS(hidden_states=None, residual=None, pdmux_pp_skip_logits=True)
    ids = torch.tensor([1, 2, 3])
    assert model.forward_split_prefill(ids, ids, batch, (0, 8)) is None


def test_reject_zero_budget_or_empty_request():
    with pytest.raises(AssertionError):
        plan_chunks([8], 0)
    with pytest.raises(AssertionError):
        plan_chunks([0], 8)


def test_sampling_uses_only_terminal_rows_and_absolute_positions():
    from sglang.srt.managers.tp_worker import TpModelWorker

    worker = TpModelWorker.__new__(TpModelWorker)
    worker.pp_group = NS(is_last_rank=True)
    metadata = NS(
        seq_lens=torch.tensor([13, 31]),
        seq_lens_cpu=torch.tensor([13, 31]),
        sampling_info=object(),
    )
    batch = NS(
        split_forward_batch=None,
        split_index=0,
        split_forward_count=4,
        pdmux_pp_final_indices=[0],
        pdmux_pp_full_prefill_tokens=32,
        seq_lens_cpu_cache=torch.tensor([13, 31]),
        get_model_worker_batch=lambda *_: metadata,
    )
    logits = NS(next_token_logits=torch.arange(8).view(2, 4))
    logits_storage = logits.next_token_logits.untyped_storage().data_ptr()
    positions_storage = metadata.seq_lens.untyped_storage().data_ptr()

    def sample(output, mwb):
        assert output.next_token_logits.tolist() == [[0, 1, 2, 3]]
        assert mwb.seq_lens.tolist() == [13]
        assert output.next_token_logits.untyped_storage().data_ptr() == logits_storage
        assert mwb.seq_lens.untyped_storage().data_ptr() == positions_storage
        return torch.tensor([7])

    worker.model_runner = NS(forward=lambda *a, **kw: (logits, False), sample=sample)
    with patch(
        "sglang.srt.managers.tp_worker.ForwardBatch.init_new", return_value=NS()
    ):
        _, ids, _, _ = worker.forward_batch_split_prefill(batch)
    assert ids.tolist() == [7]


def test_admission_fence_covers_child_gpu_metadata():
    order = []

    def record():
        event = len(order)
        order.append(("record", event))
        return event

    req = NS(rid="r", prefix_indices=[])
    parent_batch = NS(reqs=[req], extend_num_tokens=8, prepare_for_extend=lambda: None)
    s = SchedulerMultiplexPPMixin()
    s.pp_rank = 0
    s._pp_prefills = {}
    s._pp_cached_tokens = 0
    s.waiting_queue = [req]
    s.req_to_token_pool = s.token_to_kv_pool_allocator = s.tree_cache = None
    s.model_config = s.spec_algorithm = None
    s.tp_worker = NS(model_runner=NS(model=NS(start_layer=0)))
    s._pp_decode_stream = NS(record_event=record)
    s._pp_prefill_stream = NS(wait_event=lambda e: order.append(("wait", e)))
    s._pp_timing = NS(request_event=lambda *a, **kw: None)
    plan = PrefillPlan(7, [], ["r"], plan_chunks([8], 4))

    def child(*args):
        order.append(("child_metadata", None))
        return NS(reqs=[req])

    with patch(
        "sglang.srt.multiplex.multiplexing_pp.ScheduleBatch.init_new",
        return_value=parent_batch,
    ), patch("sglang.srt.multiplex.multiplexing_pp.chunk_batch", side_effect=child):
        s._pp_build_prefill(plan, [req])
    owner = s._pp_prefills[7]
    assert all(c.event == owner.event for c in owner.chunks)
    assert owner.event > max(i for i, v in enumerate(order) if v[0] == "child_metadata")
    assert order[-1] == ("wait", owner.event)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize(
    "pending,credited,length,cap,refill,budget,allowed",
    [
        (12, 0, 1, 13, 0, 16, True),
        (12, 0, 1, 12, 0, 16, False),
        (12, 4, 4, 12, 0, 16, True),
        (0, 0, 20, 8, 0, 16, True),
        (1, 0, 20, 8, 0, 16, False),
        (1, 0, 20, 64, 0, 16, True),
        # Dribbled credits cannot create a new tiny batch until refill.
        (12, 0, 1, 16, 8, 16, False),
        (12, 4, 1, 16, 8, 16, True),
        # Respect a smaller current ALP budget instead of over-waiting.
        (12, 0, 1, 16, 8, 4, True),
        # No pending work, including completed parents awaiting MERGE, must
        # still allow a single oversized FCFS head to make progress.
        (0, 0, 20, 8, 8, 16, True),
        (12, 12, 20, 8, 8, 16, True),
        # Passing refill does not relax the total token cap.
        (12, 4, 20, 16, 8, 16, False),
    ],
)
def test_token_admission_cap_and_oversized_fcfs_head(
    pending, credited, length, cap, refill, budget, allowed, adaptive
):
    from sglang.srt.multiplex.pp_admission import AdmissionStats

    s = SchedulerMultiplexPPMixin()
    s.pp_size, s._pp_max_prefills, s._pp_admission_budget, s._pp_batch_id = (
        2,
        1,
        budget,
        9,
    )
    s.max_running_requests = 128
    s.server_args = NS(
        pdmux_pp_adaptive_inflight=adaptive,
        pdmux_pp_reserve_decode=True,
        pdmux_pp_balance_merge=True,
        max_micro_batch_size=None,
        pdmux_pp_max_inflight_tokens=cap,
        pdmux_pp_token_refill_size=refill,
        pdmux_pp_prefill_chunk_size=4,
    )
    req = NS(
        rid="head",
        origin_input_ids=list(range(length)),
        fill_ids=list(range(length)),
        prefix_indices=[],
        last_node="node",
        extend_input_len=length,
        sampling_params=NS(max_new_tokens=1),
        init_next_round_input=lambda _: None,
    )
    s.waiting_queue = [req]
    s.running_mbs = [NS(reqs=[]), NS(reqs=[])]
    s._pp_pending_merges = [[], []]
    s._pp_prefills = (
        {0: NS(batch=NS(reqs=[], extend_num_tokens=pending), completed_tokens=credited)}
        if pending
        else {}
    )
    if adaptive and pending:
        # Completed parents can still own KV while waiting for MERGE. Their
        # batch count must not restore the old gate when token credits are on.
        for key in (1, 2):
            s._pp_prefills[key] = NS(
                batch=NS(reqs=[], extend_num_tokens=1), completed_tokens=1
            )
    s.mux_pp_policy = NS(merge_target=lambda _: 0)
    s.req_to_token_pool = NS(available_size=lambda: 128)
    s.token_to_kv_pool_allocator = NS(available_size=lambda: 10000)
    locks = []
    s.tree_cache = NS(
        inc_lock_ref=lambda n: locks.append(1),
        dec_lock_ref=lambda n: locks.append(-1),
        evictable_size=lambda: 0,
    )
    s._pp_admission_stats = NS(
        capacity_reason=AdmissionStats.capacity_reason, record=lambda *a, **kw: None
    )
    built = []
    s._pp_build_prefill = lambda plan, reqs: built.append(plan)
    s._pp_make_plan()
    assert bool(built) == allowed
    assert sum(locks) == int(allowed)


@pytest.mark.parametrize("append,prefix", [(None, 0), (None, 5), (False, 5), (True, 5)])
def test_append_chunk_plans_one_full_context_attention(append, prefix, monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_TORCH_COMPILE", "0")
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.is_multimodal = backend.enable_deterministic = False
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_split_tile_size = None
    plans = []
    backend.indices_updater_prefill = NS(update=lambda *a, **kw: plans.append(kw))
    batch = NS(
        forward_mode=ForwardMode.SPLIT_PREFILL,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([prefix + 4]),
        seq_lens_cpu=torch.tensor([prefix + 4]),
        seq_lens_sum=prefix + 4,
        extend_prefix_lens=torch.tensor([prefix]),
        extend_prefix_lens_cpu=[prefix],
        encoder_lens=None,
    )
    if append is not None:
        batch.pdmux_pp_append_chunk = append
    backend.init_forward_metadata(batch)
    assert plans[0]["use_ragged"] == (append is not True)
    assert plans[0]["disable_split_kv"] == (append is True)
    assert backend.forward_metadata.use_ragged == (append is not True)
