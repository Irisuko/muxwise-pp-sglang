"""Chunk coverage, shared KV ownership and terminal-only sampling regressions."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.multiplex.pp_chunk import chunk_batch, lane_works, plan_chunks
from sglang.srt.multiplex.multiplexing_pp import (
    PrefillPlan,
    PrefillWork,
    SchedulerMultiplexPPMixin,
)


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
    assert seen == [(i, j) for i, n in enumerate(lengths) for j in range(n)]
    assert finals == list(range(len(lengths)))


def test_views_preserve_full_allocation_and_request_state():
    reqs = [NS(rid=str(i), to_abort=False) for i in range(3)]
    parent = NS(
        reqs=reqs,
        extend_lens=[3, 9, 2],
        prefix_lens=[2, 7, 0],
        input_ids=torch.arange(14),
        out_cache_loc=torch.arange(40, 54),
        req_pool_indices=torch.tensor([5, 2, 9]),
        orig_seq_lens=torch.tensor([5, 16, 2]),
        multimodal_inputs=[None] * 3,
        model_config=NS(vocab_size=100),
        device="cpu",
    )
    with patch(
        "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
        side_effect=lambda batch, _: [r.rid for r in batch.reqs],
    ):
        children = [chunk_batch(parent, c) for c in plan_chunks(parent.extend_lens, 4)]
    assert [b.prefix_lens for b in children] == [[2, 7], [8], [12], [0]]
    assert [b.seq_lens.tolist() for b in children] == [[5, 8], [12], [16], [2]]
    assert [b.sampling_info for b in children] == [["0"], None, ["1"], ["2"]]
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
