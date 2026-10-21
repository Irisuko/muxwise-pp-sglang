"""CPU checks for PP split boundaries, cache agreement, and nonblocking progress."""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.multiplex.multiplexing_pp import (
    DeterministicRadixFIFO,
    OrderedRadixEviction,
    PrefillMailbox,
    PrefillPlan,
    PrefillReceivePool,
    PrefillWork,
    SchedulerMultiplexPPMixin,
    StaticMuxPPPolicy,
)
from sglang.srt.multiplex.pdmux_context import PDMuxConfig


class FakeLayer(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.number = number

    def forward(self, positions, hidden, batch, residual):
        residual = hidden if residual is None else residual + hidden
        return hidden + self.number, residual


class TestSplitPrefill(unittest.TestCase):
    def model(self, start, end, total=8):
        obj = LlamaForCausalLM.__new__(LlamaForCausalLM)
        nn.Module.__init__(obj)
        obj.pp_group = SimpleNamespace(
            is_first_rank=start == 0, is_last_rank=end == total
        )

        def forbidden(*args):
            self.fail("Embedding or norm executed on the wrong stage")

        obj.model = SimpleNamespace(
            start_layer=start,
            end_layer=end,
            config=SimpleNamespace(num_hidden_layers=total),
            layers=[
                FakeLayer(i + 1) if start <= i < end else None for i in range(total)
            ],
            embed_tokens=(
                (lambda ids: ids.float().unsqueeze(-1)) if start == 0 else forbidden
            ),
            norm=(
                (lambda hidden, residual: (hidden + residual, None))
                if end == total
                else forbidden
            ),
        )
        obj.lm_head = None
        obj.logits_processor = lambda ids, hidden, head, batch: hidden
        return obj

    def test_stage_boundaries_and_residual(self):
        ids = torch.tensor([3, 9, 5])
        whole = SimpleNamespace(hidden_states=None, residual=None)
        expected = self.model(0, 8).forward_split_prefill(ids, ids, whole, (0, 8))
        proxy = None
        for start in range(0, 8, 2):
            stage = self.model(start, start + 2)
            batch = SimpleNamespace(hidden_states=None, residual=None)
            self.assertIsNone(
                stage.forward_split_prefill(
                    ids, ids, batch, (start, start + 1), pp_proxy_tensors=proxy
                )
            )
            proxy = stage.forward_split_prefill(ids, ids, batch, (start + 1, start + 2))
        torch.testing.assert_close(expected, proxy)

    def test_reject_foreign_layer_interval(self):
        with self.assertRaises(AssertionError):
            self.model(2, 4).forward_split_prefill(None, None, None, (0, 2))


class TestProtocol(unittest.TestCase):
    def test_prefetched_ready_dispatches_without_scheduler_progress(self):
        mailbox = PrefillMailbox.__new__(PrefillMailbox)
        task = (True, {"hidden_states": object()}, object(), object(), object())
        mailbox.pending_sends = {12: task}
        mailbox.send_lock = threading.Lock()
        mailbox.transfers, mailbox.inbox = Queue(), Queue()
        mailbox._dispatch_ready(12)
        self.assertIs(mailbox.transfers.get_nowait(), task)
        self.assertEqual(mailbox.pending_sends, {})
        self.assertTrue(mailbox.inbox.empty())
        mailbox._dispatch_ready(13)
        self.assertEqual(mailbox.inbox.get_nowait(), dict(kind="ready", batch_id=13))

    def test_receive_slots_bound_memory_and_preserve_live_inputs(self):
        pool = PrefillReceivePool(16, 4, torch.float32, "cpu")
        specs = [
            (name, (8, 4), torch.float32) for name in ("hidden_states", "residual")
        ]
        first, a = pool.acquire(specs)
        second, b = pool.acquire(specs)
        a["residual"].fill_(7)
        b["residual"].fill_(9)
        self.assertTrue(torch.all(a["residual"] == 7))
        self.assertIsNone(pool.acquire(specs))
        with self.assertRaises(AssertionError):
            pool.acquire([(name, (17, 4), dtype) for name, _, dtype in specs])
        pool.release(second)
        reused, c = pool.acquire(specs)
        self.assertEqual(second, reused)
        self.assertEqual(b["hidden_states"].data_ptr(), c["hidden_states"].data_ptr())
        self.assertTrue(torch.all(a["residual"] == 7))
        pool.release(first)
        with self.assertRaises(AssertionError):
            pool.release(first)
        pool.release(reused)
        self.assertEqual(len(pool.free), 2)

    def test_merge_releases_slot_even_before_send_polling(self):
        s = SchedulerMultiplexPPMixin()
        s._pp_timing = SimpleNamespace(request_event=lambda *a, **kw: None)
        s.tp_size, s.tp_rank, s._pp_leader = 1, 0, False
        s.device, s.pp_rank = "cpu", 1
        waits = []
        s._pp_decode_stream = SimpleNamespace(
            wait_event=lambda e: waits.append(("decode", e))
        )
        s._pp_receive_stream = SimpleNamespace(
            wait_event=lambda e: waits.append(("receive", e))
        )
        s._pp_receive_pool = PrefillReceivePool(8, 4, torch.float32, "cpu")
        slot, _ = s._pp_receive_pool.acquire(
            [(k, (4, 4), torch.float32) for k in ("hidden_states", "residual")]
        )
        batch = SimpleNamespace(
            split_prefill_finished=True,
            filter_batch=lambda: None,
            is_empty=lambda: True,
        )
        event, launched = object(), threading.Event()
        launched.set()
        work = PrefillWork(
            PrefillPlan(7, [], ["r"]), batch, "sending", event, launched, recv_slot=slot
        )
        s._pp_prefills = {7: work}
        s._pp_pending_merges, s.pp_size = [[], []], 2
        s.split_prefill_batch = batch
        s.process_batch_result = lambda batch, result: None
        s._pp_merge(dict(batch_id=7, target_mb=0, next_token_ids=[9], targets=[0]), 0)
        self.assertEqual(waits, [("decode", event), ("receive", event)])
        self.assertEqual(len(s._pp_receive_pool.free), 2)
        self.assertIsNone(work.recv_slot)
        self.assertEqual(s._pp_prefills, {})

    def test_abort_covers_all_lanes_and_emits_terminal_output(self):
        scheduler = SchedulerMultiplexPPMixin()

        def req(rid):
            return SimpleNamespace(
                rid=rid, return_logprob=False, finished=lambda: False, to_abort=False
            )

        queued, decode0, decode1, prefill = [req(rid) for rid in ("q", "d0", "d1", "p")]
        scheduler.waiting_queue = [queued]
        scheduler._pp_pending_merges = []
        scheduler.running_mbs = [
            SimpleNamespace(reqs=[decode0]),
            SimpleNamespace(reqs=[decode1]),
        ]
        scheduler._pp_prefills = {
            0: SimpleNamespace(batch=SimpleNamespace(reqs=[prefill]))
        }
        terminal = []
        scheduler.stream_output = lambda reqs, _: terminal.extend(reqs)
        scheduler._pp_apply_aborts(["q", "d0", "d1", "p"])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(terminal, [queued])
        self.assertEqual(queued.finished_reason.to_json()["type"], "abort")
        self.assertTrue(all(r.to_abort for r in (decode0, decode1, prefill)))

    def test_pending_nccl_host_launch_never_synchronizes(self):
        s = SchedulerMultiplexPPMixin()
        s._pp_timing = SimpleNamespace(request_event=lambda *a, **kw: None)
        s.tp_size, s.tp_rank, s._pp_leader = 1, 0, False
        s._pp_timing = SimpleNamespace(
            prefill_state=lambda: None, request_event=lambda *a, **kw: None
        )
        s.prefill_mailbox = SimpleNamespace(poll=lambda: None)
        work = PrefillWork(None, None, "receiving", launched=threading.Event())
        s._pp_prefills = {0: work}

        class UnrecordedEvent:
            def query(self):
                raise AssertionError("Unrecorded event must not be queried")

        work.event = UnrecordedEvent()
        s._pp_progress_prefill()
        s.mux_pp_policy = StaticMuxPPPolicy()
        s.pdmux_config = PDMuxConfig(manual_divisions=[[96, 32, 1]])
        s.stream_groups = [None] * 3
        s._pp_stream_idx = 0
        s._pp_adjust_streams(8)
        self.assertEqual(s._pp_stream_idx, 0)

    def test_local_fifo_skips_sent_batches_but_keeps_their_kv_owner(self):
        s = SchedulerMultiplexPPMixin()
        s._pp_timing = SimpleNamespace(request_event=lambda *a, **kw: None)
        s.tp_size, s.tp_rank, s._pp_leader = 1, 0, False
        first = PrefillWork(None, object(), "done")
        second = PrefillWork(SimpleNamespace(batch_id=2), object(), "waiting")
        third = PrefillWork(SimpleNamespace(batch_id=3), object(), "waiting")
        s._pp_prefills = {1: first, 2: second, 3: third}
        s._pp_envelopes = {}
        s._pp_timing = SimpleNamespace(
            prefill_state=lambda: None, request_event=lambda *a, **kw: None
        )
        s.prefill_mailbox = SimpleNamespace(poll=lambda: None)
        s._pp_progress_prefill()
        self.assertIs(s.split_prefill_batch, second.batch)
        self.assertEqual(len(s._pp_prefills), 3)
        self.assertEqual(third.state, "waiting")

    def test_partition_and_least_loaded_hooks(self):
        p = StaticMuxPPPolicy()
        cfg = PDMuxConfig(manual_divisions=[[96, 32, 1], [80, 48, 8], [64, 64, 16]])
        self.assertEqual(
            [p.partition(cfg, 5, bs, True) for bs in [0, 1, 8, 16]], [0, 1, 2, 3]
        )
        self.assertEqual(p.partition(cfg, 5, 1, False), 4)
        req = SimpleNamespace(finished=lambda: False)
        finished = SimpleNamespace(finished=lambda: True)
        mbs = [SimpleNamespace(reqs=rs) for rs in [[req, req], [req], [finished], []]]
        self.assertEqual(p.merge_target(mbs), 2)
        self.assertEqual(
            p.slice_layers(PDMuxConfig(split_forward_token_budget=8192), 8000, 8), 1
        )

    def test_radix_eviction_ignores_stage_wall_clock(self):
        caches = []
        for offset in [0, 100]:
            freed = []
            pool = SimpleNamespace(
                device="cpu",
                free=lambda value, freed=freed: freed.extend(value.tolist()),
            )
            cache = RadixCache(None, pool, 1)
            cache.eviction_strategy = DeterministicRadixFIFO()
            cache.insert(RadixKey([1, 2, 3, 4]), torch.arange(4) + offset)
            cache.insert(RadixKey([1, 2, 5, 6]), torch.arange(4) + offset + 10)
            # A rejected admission probe still splits the tree, and must be replayed.
            self.assertEqual(
                len(cache.match_prefix(RadixKey([1, 2, 3])).device_indices), 3
            )
            caches.append((cache, freed, offset))
        for i, (cache, freed, offset) in enumerate(caches):
            for n, node in enumerate(cache._collect_leaves()):
                node.last_access_time = n if i == 0 else -n
            cache.evict(2)
        self.assertEqual(caches[0][1], [x - 100 for x in caches[1][1]])
        for key in [[1, 2, 3, 4], [1, 2, 5, 6]]:
            lengths = [
                len(c.match_prefix(RadixKey(key)).device_indices) for c, _, _ in caches
            ]
            self.assertEqual(lengths[0], lengths[1])


class TestOrderedLRU(unittest.TestCase):
    def test_lru_victims_replayed_with_different_clocks_and_ids(self):
        caches = []
        for rank in range(2):
            freed = []
            cache = RadixCache(
                None,
                SimpleNamespace(
                    device="cpu", free=lambda v, f=freed: f.extend(v.tolist())
                ),
                1,
            )
            for key in ([1, 2, 3], [1, 2, 4], [5, 6]):
                cache.insert(RadixKey(key), torch.tensor(key) + rank * 100)
            for n, node in enumerate(cache._collect_leaves()):
                node.last_access_time = n if rank == 0 else -n
            caches.append((cache, freed, OrderedRadixEviction(cache, rank == 0)))
        commands = []
        for cache, freed, journal in caches:
            journal.begin(commands)
            cache.evict(3)
            journal.end()
        self.assertEqual(caches[0][1], [v - 100 for v in caches[1][1]])
        self.assertTrue(commands[0][1])
        for key in ([1, 2, 3], [1, 2, 4], [5, 6]):
            self.assertEqual(
                *[
                    len(c.match_prefix(RadixKey(key)).device_indices)
                    for c, _, _ in caches
                ]
            )


if __name__ == "__main__":
    unittest.main()
