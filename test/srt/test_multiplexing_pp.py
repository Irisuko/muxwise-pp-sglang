"""CPU checks for PP split boundaries, cache agreement, and nonblocking progress."""

import threading
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.multiplex.multiplexing_pp import (
    DeterministicRadixFIFO,
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
    def test_abort_covers_all_lanes_and_emits_terminal_output(self):
        scheduler = SchedulerMultiplexPPMixin()

        def req(rid):
            return SimpleNamespace(
                rid=rid, return_logprob=False, finished=lambda: False, to_abort=False
            )

        queued, decode0, decode1, prefill = [req(rid) for rid in ("q", "d0", "d1", "p")]
        scheduler.waiting_queue = [queued]
        scheduler.running_mbs = [
            SimpleNamespace(reqs=[decode0]),
            SimpleNamespace(reqs=[decode1]),
        ]
        scheduler.split_prefill_batch = SimpleNamespace(reqs=[prefill])
        terminal = []
        scheduler.stream_output = lambda reqs, _: terminal.extend(reqs)
        scheduler._pp_apply_aborts(["q", "d0", "d1", "p"])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(terminal, [queued])
        self.assertEqual(queued.finished_reason.to_json()["type"], "abort")
        self.assertTrue(all(r.to_abort for r in (decode0, decode1, prefill)))

    def test_pending_nccl_host_launch_never_synchronizes(self):
        s = SchedulerMultiplexPPMixin()
        s.prefill_mailbox = SimpleNamespace(poll=lambda: None)
        s._pp_prefill_state = "receiving"
        s._pp_io_launched = threading.Event()

        class UnrecordedEvent:
            def query(self):
                raise AssertionError("Unrecorded event must not be queried")

        s._pp_prefill_event = UnrecordedEvent()
        s._pp_progress_prefill()
        s.mux_pp_policy = StaticMuxPPPolicy()
        s.pdmux_config = PDMuxConfig(manual_divisions=[[96, 32, 1]])
        s.stream_groups = [None] * 3
        s._pp_stream_idx = 0
        s._pp_adjust_streams(8)
        self.assertEqual(s._pp_stream_idx, 0)

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


if __name__ == "__main__":
    unittest.main()
