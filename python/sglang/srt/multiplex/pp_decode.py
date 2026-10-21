"""Prefetched PP controls and asynchronous, fixed-shape decode P2P.

Control(step, mb) is the only forward CPU envelope, including proxy batch size
and the preceding result for that slot. Last-stage tokens take one pinned D2H
copy, return to stage0, and are replayed in that envelope on every stage.
Only the scheduler mutates requests/cache; transport threads own no such state.

Like upstream SGLang's P2PWork, each send retains both Work and payload until
buffer reuse. Receive buffers are posted ahead by the control receiver and may
be reused only after the scheduler records consumption. CUDA-graph outputs are
copied to owned slots before another graph replay can overwrite them.
"""

import pickle
import queue
import threading
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class P2PWork:
    payload: dict
    works: list = field(default_factory=list)
    launched: threading.Event = field(default_factory=threading.Event)

    def wait(self, check=lambda: None):
        while not self.launched.wait(timeout=0.1):
            check()
        for work in self.works:
            work.wait()
        # NCCL wait orders the current CUDA stream, not the scheduler CPU.
        self.works.clear()


class AsyncDecodeTransport:
    def __init__(self, group, max_bs, hidden_size, dtype, device, token_leader=True):
        self.group, self.rank, self.size = group, group.rank_in_group, group.world_size
        # CUDA device selection is thread-local; an unindexed "cuda" in a
        # new transport thread would otherwise select GPU0 on every PP rank.
        self.device, self.max_bs = getattr(group, "device", device), max_bs
        self.errors = queue.Queue()
        # Empty control turns have no token-ring backpressure. Bound them too,
        # so idle traffic cannot build a backlog ahead of newly arrived work.
        self.controls_in, self.controls_out = queue.Queue(self.size), queue.Queue(
            self.size
        )
        self.tokens_in, self.tokens_out = queue.Queue(self.size), queue.Queue(self.size)
        self.sends = queue.Queue(self.size)
        self.rx_stream, self.tx_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self.rx_slots, self.tx_slots = [], []
        self.rx_free, self.rx_consumed = [], [None] * self.size
        self.tx_work = [None] * self.size
        self.token_free, self.token_slots = [], []
        for _ in range(self.size):

            def buffers():
                return {
                    name: torch.empty((max_bs, hidden_size), dtype=dtype, device=device)
                    for name in ("hidden_states", "residual")
                }

            if self.rank:
                self.rx_slots.append(buffers())
                available = threading.Event()
                available.set()
                self.rx_free.append(available)
            if self.rank < self.size - 1:
                self.tx_slots.append(buffers())
            else:
                self.token_slots.append(
                    torch.empty(max_bs, dtype=torch.int64, pin_memory=True)
                )
                available = threading.Event()
                available.set()
                self.token_free.append(available)
        # Allocate all slots before any unmatched receive can be posted.
        for fn, enabled in (
            (self._receive_controls, self.rank > 0),
            (self._send_controls, self.rank < self.size - 1),
            (self._send_proxies, self.rank < self.size - 1),
            (self._receive_tokens, self.rank == 0 and token_leader),
            (self._send_tokens, self.rank == self.size - 1 and token_leader),
        ):
            if enabled:
                threading.Thread(target=self._guard, args=(fn,), daemon=True).start()

    def _guard(self, fn):
        try:
            torch.cuda.set_device(self.device)
            with torch.inference_mode():
                fn()
        except BaseException as exc:
            self.errors.put(exc)

    def check(self):
        if not self.errors.empty():
            raise RuntimeError(
                "Asynchronous PP decode transport failed"
            ) from self.errors.get()

    def _wait_flag(self, flag):
        while not flag.wait(timeout=0.1):
            self.check()

    def _get(self, inbox):
        while True:
            self.check()
            try:
                return inbox.get(timeout=0.1)
            except queue.Empty:
                pass

    def _put(self, outbox, value):
        while True:
            self.check()
            try:
                outbox.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def send_control(self, packet):
        # Freeze mutable Req/control state on the scheduler before handing it off.
        self._put(
            self.controls_out, pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def _send_controls(self):
        while (payload := self.controls_out.get()) is not None:
            dist.send_object_list(
                [payload],
                dst=self.group.ranks[self.rank + 1],
                group=self.group.cpu_group,
            )

    def _receive_controls(self):
        while True:
            obj = [None]
            dist.recv_object_list(
                obj, src=self.group.ranks[self.rank - 1], group=self.group.cpu_group
            )
            packet = pickle.loads(obj[0])
            transfer = None
            if packet.decode_bs:
                mb, bs = packet.mb_id, packet.decode_bs
                assert 0 < bs <= self.max_bs
                self.rx_free[mb].wait()
                self.rx_free[mb].clear()
                tensors = {k: t[:bs] for k, t in self.rx_slots[mb].items()}
                transfer = P2PWork(tensors)
                with torch.cuda.stream(self.rx_stream):
                    if self.rx_consumed[mb] is not None:
                        self.rx_stream.wait_event(self.rx_consumed[mb])
                    transfer.works = [
                        dist.irecv(
                            t,
                            src=self.group.ranks[self.rank - 1],
                            group=self.group.device_group,
                        )
                        for t in tensors.values()
                    ]
                transfer.launched.set()
            self.controls_in.put((packet, transfer))

    def receive_control(self):
        return self._get(self.controls_in)

    def release_proxy(self, mb):
        self.rx_consumed[mb] = torch.cuda.current_stream().record_event()
        self.rx_free[mb].set()

    def send_proxy(self, mb, tensors):
        if self.tx_work[mb] is not None:
            self.tx_work[mb].wait(self.check)  # Only before this slot is overwritten.
        bs = tensors["hidden_states"].shape[0]
        assert 0 < bs <= self.max_bs and set(tensors) == set(self.tx_slots[mb])
        payload = {k: t[:bs] for k, t in self.tx_slots[mb].items()}
        for key, dest in payload.items():
            dest.copy_(tensors[key])
        ready = torch.cuda.current_stream().record_event()
        transfer = self.tx_work[mb] = P2PWork(payload)
        self._put(self.sends, (transfer, ready))

    def _send_proxies(self):
        while (job := self.sends.get()) is not None:
            transfer, ready = job
            with torch.cuda.stream(self.tx_stream):
                self.tx_stream.wait_event(ready)
                transfer.works = [
                    dist.isend(
                        t,
                        dst=self.group.ranks[self.rank + 1],
                        group=self.group.device_group,
                    )
                    for t in transfer.payload.values()
                ]
            transfer.launched.set()

    def send_tokens(self, step, mb, tokens, can_run_cuda_graph):
        self._wait_flag(
            self.token_free[mb]
        )  # The CPU sender must have read the old slot.
        self.token_free[mb].clear()
        bs = tokens.numel()
        self.token_slots[mb][:bs].copy_(tokens, non_blocking=True)
        ready = torch.cuda.current_stream().record_event()
        self._put(self.tokens_out, (step, mb, bs, can_run_cuda_graph, ready))

    def _send_tokens(self):
        while (job := self.tokens_out.get()) is not None:
            step, mb, bs, graph, ready = job
            ready.synchronize()  # Off the scheduler; one D2H for all PP stages.
            packet = dict(
                step=step,
                mb_id=mb,
                tokens=self.token_slots[mb][:bs].tolist(),
                can_run_cuda_graph=graph,
            )
            dist.send_object_list(
                [packet], dst=self.group.ranks[0], group=self.group.cpu_group
            )
            self.token_free[mb].set()

    def _receive_tokens(self):
        while True:
            obj = [None]
            dist.recv_object_list(
                obj, src=self.group.ranks[-1], group=self.group.cpu_group
            )
            self.tokens_in.put(obj[0])

    def receive_tokens(self, step, mb):
        packet = self._get(self.tokens_in)
        assert (packet["step"], packet["mb_id"]) == (
            step,
            mb,
        ), "PP result order mismatch"
        return packet

    def close(self):
        for outbox in (self.controls_out, self.sends, self.tokens_out):
            try:
                outbox.put_nowait(None)
            except queue.Full:
                pass  # Never block exception cleanup on a failed peer/thread.
