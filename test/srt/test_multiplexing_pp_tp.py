"""Real CPU/Gloo TP controls and asymmetric completion, without CUDA workers."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import pipeline_group_ranks
from sglang.srt.multiplex.multiplexing_pp import (
    MigrationPlan,
    PPControl,
    PrefillPlan,
    SchedulerMultiplexPPMixin,
)


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    s = SchedulerMultiplexPPMixin()
    s.server_args = SimpleNamespace(pdmux_pp_runnable_partition=False)
    s.tp_size, s.tp_rank, s.pp_rank = 2, rank % 2, rank // 2
    s.tp_cpu_group = groups[s.pp_rank]
    s.tp_group = SimpleNamespace(ranks=[2 * s.pp_rank, 2 * s.pp_rank + 1])
    expected = PPControl(
        17,
        1,
        plan=PrefillPlan(8, [("a", 6, 2)], ["a"]),
        abort=["b"],
        retract=["c"],
        migrations=[MigrationPlan(["d"], 0, 1)],
        evictions=[(4, [(None, [1, 2, 3, 4])])],
        required=123.5,
        admission_budget=4096,
        merge={"batch_id": 7, "next_token_ids": [42]},
        retired={"step": 15, "mb_id": 1, "tokens": [6]},
        decode_notice={
            "step": 17,
            "mb": 1,
            "bs": 4,
            "ctx": 8192,
            "issued_at": 1.0,
            "eta": 1.01,
            "seconds": 0.01,
            "host": "node",
        },
    )
    packet = (
        s._pp_tp_broadcast(expected if rank == 0 else None) if s.pp_rank == 0 else None
    )
    # Same-rank PP paths: no rank is omitted from the control chain.
    if s.pp_rank == 0:
        dist.send_object_list([packet], dst=rank + 2)
    else:
        obj = [None]
        dist.recv_object_list(obj, src=rank - 2)
        packet = obj[0]
    assert packet == expected
    s.waiting_queue = [
        SimpleNamespace(
            rid="a",
            fill_ids=list(range(6)),
            prefix_indices=[1, 2],
            last_node="node",
            init_next_round_input=lambda _: None,
        )
    ]
    locked, built = [], []
    s.tree_cache = SimpleNamespace(inc_lock_ref=locked.append)
    s._pp_build_prefill = lambda plan, reqs: built.append(
        (plan.batch_id, [r.rid for r in reqs])
    )
    s._pp_apply_plan(packet.plan)
    assert locked == ["node"] and built == [(8, ["a"])]
    # Rank 1 is behind: neither peer may advance its prefill or change SMs.
    assert not s._pp_tp_all(s.tp_rank == 0)
    assert s._pp_tp_all(True)

    class Event:
        def query(self):
            return s.tp_rank == 0

    s._pp_prefills = {8: SimpleNamespace(state="compute", launched=None, event=Event())}
    s._pp_stream_idx = 0
    s.pdmux_config = None
    s.stream_groups = [None] * 3

    def choose(*_):
        assert s.tp_rank == 0, "Follower must never run partition policy"
        return 2

    s.mux_pp_policy = SimpleNamespace(partition=choose)
    switches, syncs = [], []
    s._pp_select_streams = switches.append
    s._pp_timing = SimpleNamespace(region=lambda *_: nullcontext())
    s._pp_prefill_stream = s._pp_decode_stream = SimpleNamespace(
        synchronize=lambda: syncs.append(True)
    )
    s.sm_counts = [(128, 0), (64, 64), (0, 128)]
    s._pp_adjust_streams(1)
    assert not switches and not syncs
    s._pp_prefills[8].event = None
    s._pp_adjust_streams(1)
    assert switches == [2] and len(syncs) == 2
    dist.destroy_process_group()


def test_tp_control_replay_and_partition_consensus(tmp_path):
    mp.start_processes(
        _worker,
        args=(f"file://{tmp_path}/gloo",),
        nprocs=4,
        start_method="fork",
        join=True,
    )


def test_pipeline_pairing():
    assert pipeline_group_ranks(4, 2) == [[0, 2], [1, 3]]
    assert pipeline_group_ranks(8, 4) == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert pipeline_group_ranks(4, 4) == [[0, 1, 2, 3]]


def test_tp1_has_no_added_collectives(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("TP1 must not issue TP collectives")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    monkeypatch.setattr(dist, "broadcast_object_list", forbidden)
    s = SchedulerMultiplexPPMixin()
    s.server_args = SimpleNamespace(pdmux_pp_runnable_partition=False)
    s.tp_size = 1
    packet = PPControl(3, 1)
    assert s._pp_tp_broadcast(packet) is packet
    assert s._pp_tp_all(True)
    assert not s._pp_tp_all(False)


def test_server_args_accept_tp_and_reject_dp_attention(monkeypatch):
    import pytest
    from sglang.srt.server_args import ServerArgs

    # Test validation without hardware probing/model loading in __post_init__.
    monkeypatch.setattr(ServerArgs, "__post_init__", lambda self: None)
    args = ServerArgs(
        model_path="llama",
        tp_size=2,
        pp_size=2,
        enable_pdmux=True,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        pdmux_prefill_mlp_chunk_tokens=4096,
    )
    args.check_server_args()
    args.enable_dp_attention = True
    with pytest.raises(AssertionError, match="DP attention"):
        args.check_server_args()
    args.pdmux_prefill_mlp_chunk_tokens = 0
    with pytest.raises(AssertionError, match="DP attention"):
        args.check_server_args()


def test_pp_optimization_defaults_and_independent_cli_ablations():
    import argparse
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "llama"])
    enabled = ["latency_partition"]
    disabled = [
        "cpu_merge_tokens",
        "lazy_admission_budget",
        "runnable_partition",
        "decode_aware_slices",
        "adaptive_inflight",
    ]
    for name in enabled:
        assert getattr(args, "pdmux_pp_" + name) is True
    for name in disabled:
        assert getattr(args, "pdmux_pp_" + name) is False
    flags = ["--no-pdmux-pp-" + x.replace("_", "-") for x in enabled]
    flags += ["--pdmux-pp-" + x.replace("_", "-") for x in disabled]
    args = parser.parse_args(["--model-path", "llama"] + flags)
    for name in enabled:
        assert getattr(args, "pdmux_pp_" + name) is False
    for name in disabled:
        assert getattr(args, "pdmux_pp_" + name) is True
