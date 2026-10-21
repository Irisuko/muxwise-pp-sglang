"""Real four-process Gloo ordering/reuse test of the decode transport, no GPU.

CUDA streams/events are immediate CPU stand-ins here; NCCL/graph dependencies
are separately covered by the GPU serial, cancellation, and saturation tests.
"""

import contextlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.multiplex.pp_decode import AsyncDecodeTransport, P2PWork


class CPUEvent:
    def synchronize(self):
        pass


class CPUStream:
    def wait_event(self, event):
        pass

    def record_event(self):
        return CPUEvent()


def _rank_test(rank, path, tp_size=1):
    torch.set_num_threads(1)
    torch.cuda.set_device = lambda device: None
    torch.cuda.Stream = CPUStream
    torch.cuda.stream = lambda stream: contextlib.nullcontext()
    torch.cuda.current_stream = CPUStream
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    torch.empty = cpu_empty
    dist.init_process_group(
        "gloo", init_method="file://" + path + "/init", rank=rank, world_size=4
    )
    size, pp_rank, tp_rank = 4 // tp_size, rank // tp_size, rank % tp_size
    cpu_group = data_group = tp_group = None
    for t in range(tp_size):
        ranks = list(range(t, 4, tp_size))
        cpu, data = dist.new_group(ranks, backend="gloo"), dist.new_group(
            ranks, backend="gloo"
        )
        if rank in ranks:
            cpu_group, data_group = cpu, data
    for stage in range(size):
        ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
        cpu = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            tp_group = cpu
    group = SimpleNamespace(
        rank_in_group=pp_rank,
        world_size=size,
        ranks=list(range(tp_rank, 4, tp_size)),
        cpu_group=cpu_group,
        device_group=data_group,
    )
    transport = AsyncDecodeTransport(
        group, 4, 8, torch.float32, "cpu", token_leader=tp_rank == 0
    )
    flight = [None] * size
    returns = 0
    for step in range(52):
        mb = step % size
        bs = (0 if step % 7 == 0 else step % 4 + 1) if step < 48 else 0
        if pp_rank == 0:
            retired = (
                transport.receive_tokens(flight[mb], mb)
                if flight[mb] is not None and tp_rank == 0
                else None
            )
            packet = SimpleNamespace(step=step, mb_id=mb, decode_bs=bs, retired=retired)
            if tp_size > 1:
                obj = [packet]
                dist.broadcast_object_list(obj, src=0, group=tp_group)
                packet = obj[0]
        else:
            packet, received = transport.receive_control()
            assert (packet.step, packet.mb_id, packet.decode_bs) == (step, mb, bs)
        if packet.retired:
            assert packet.retired["step"] == flight[mb]
            old = flight[mb]
            assert packet.retired["tokens"] == [
                old + sum(i * 10 for i in range(size))
            ] * (old % 4 + 1)
            returns += 1
        else:
            assert flight[mb] is None
        if pp_rank < size - 1:
            transport.send_control(packet)
        if bs:
            if pp_rank == 0:
                tensors = {
                    k: torch.full((bs, 8), float(step))
                    for k in ("hidden_states", "residual")
                }
            else:
                received.wait()
                expected = step + sum(i * 10 for i in range(pp_rank))
                assert torch.all(received.payload["hidden_states"] == expected)
                tensors = {k: t + pp_rank * 10 for k, t in received.payload.items()}
                transport.release_proxy(mb)
            if pp_rank < size - 1:
                transport.send_proxy(mb, tensors)
            elif tp_rank == 0:
                transport.send_tokens(
                    step, mb, tensors["hidden_states"][:, 0].to(torch.int64), True
                )
        flight[mb] = step if bs else None
    assert not any(v is not None for v in flight)
    assert returns == sum(i % 7 != 0 for i in range(48))
    dist.barrier()
    Path(path, str(rank) + ".json").write_text(json.dumps(dict(returns=returns)))
    # Receive threads intentionally stay posted until the serving process exits.
    os._exit(0)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_four_rank_prefetch_slots_and_retired_token_order(tmp_path, tp_size):
    processes = [
        mp.get_context("spawn").Process(
            target=_rank_test, args=(rank, str(tmp_path), tp_size)
        )
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            assert (
                process.exitcode == 0
            ), f"Transport process failed or stalled: {process.exitcode}"
        assert len(list(tmp_path.glob("*.json"))) == 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=2)


def test_p2p_retains_payload_until_wait():
    payload = {"hidden_states": object()}
    calls = []
    transfer = P2PWork(payload, [SimpleNamespace(wait=lambda: calls.append("wait"))])
    transfer.launched.set()
    assert transfer.payload is payload and not calls
    transfer.wait()
    assert calls == ["wait"] and transfer.works == []
