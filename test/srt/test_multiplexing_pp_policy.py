"""CPU tests: online predictions and ownership-preserving batch migration."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch, global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_pp import (
    SchedulerMultiplexPPMixin,
    StaticMuxPPPolicy,
)
from sglang.srt.multiplex.pp_policy import (
    ALPMuxPPPolicy,
    OnlineRLS,
    balanced_assignment,
    choose_partition,
    migration_rids,
    rebalance_pair,
    required_throughput,
    sm_features,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams


def request(i, context=100):
    return SimpleNamespace(
        rid=str(i),
        origin_input_ids=[1] * context,
        output_ids=[i + 1],
        finished=lambda: False,
        return_logprob=False,
        stream=True,
        grammar=None,
        sampling_params=SamplingParams(
            temperature=0,
            frequency_penalty=0.1 * (i % 3),
            presence_penalty=0.2 * (i % 2),
        ),
        custom_logit_processor=None,
    )


class TestPolicy(unittest.TestCase):
    def test_rls_converges_and_clips_outliers(self):
        rng = np.random.default_rng(42)
        model = OnlineRLS(3, 2)
        theta = np.array([0.002, 0.0003, 0.001])
        for i in range(700):
            x = np.r_[rng.uniform(1, 16), rng.uniform(1, 40), 1]
            model.update(x, x @ theta * (1 + rng.normal(0, 0.01)))
            if i == 14:
                self.assertFalse(model.ready)
        self.assertTrue(model.ready)
        x = np.array([7, 19, 1])
        self.assertLess(abs(model.predict(x) / (x @ theta) - 1), 0.03)
        before = model.predict(x)
        model.update(x, before * 10000)
        self.assertLess(abs(model.predict(x) - before), before * 0.4)
        model.update(x, float("nan"))
        self.assertEqual(model.count, 701)

    def test_partitions_slo_priority_and_best_effort(self):
        sm = [(128, 0), (96, 32), (80, 48), (64, 64), (0, 128)]
        td = [0, 0.03, 0.02, 0.01, 0.005]
        rates = [0, 3000, 2000, 1000, 0]
        self.assertEqual(choose_partition(sm, td, rates, 900, 4, 0.1), 3)
        self.assertEqual(choose_partition(sm, td, rates, 1500, 4, 0.1), 2)
        self.assertEqual(choose_partition(sm, td, rates, float("inf"), 4, 0.1), 2)
        self.assertEqual(choose_partition(sm, td, rates, 900, 4, 0.001), 3)
        self.assertEqual(sm_features([2, 3, 1], 64, 128), [2, 3, 1, 2, 3])

    def test_exact_demand_formula_and_deadline(self):
        self.assertAlmostEqual(
            required_throughput([(8, 100), (9.5, 20)], [(9, 200)], 10, 5, 0.1),
            100 / 3 + 20,
        )
        self.assertEqual(required_throughput([(4, 100)], [], 10, 5, 0.1), float("inf"))
        self.assertEqual(required_throughput([], [(0, 100), (9, 200)], 20, 5, 0.1), 0)

    def test_weighted_assignment_hysteresis_and_no_overshoot(self):
        batches = [[request(0, 20000)], [request(1, 10)], []]
        assigned = balanced_assignment(
            [request(i) for i in range(2, 6)], batches, [[], [], []], 1, 0.0001, 4
        )
        self.assertEqual(assigned[0], 2)
        heavy = [request(i) for i in range(10)]
        self.assertIsNone(rebalance_pair([heavy, []], 1, 0, 8))
        self.assertEqual(rebalance_pair([heavy, []], 1, 0, 1), (0, 1))
        self.assertEqual(len(migration_rids(heavy, [], 1, 0, 1, 32)), 5)
        self.assertEqual(len(migration_rids(heavy, [], 1, 0, 1, 2)), 2)
        self.assertEqual(
            migration_rids([request(1, 30000)], [request(2)], 1, 0.001, 0.1, 32), []
        )


class TestALPTransitions(unittest.TestCase):
    def policy(self):
        args = SimpleNamespace(
            pdmux_pp_slo_ttft=5,
            pdmux_pp_slo_tpot=0.1,
            pdmux_pp_tpot_coeff=1,
            pdmux_pp_min_residency_steps=16,
            pdmux_pp_hysteresis_steps=4,
            pdmux_pp_alp_admission=True,
            pdmux_pp_min_budget=1024,
            pdmux_pp_max_budget=16384,
        )
        s = SimpleNamespace(
            server_args=args,
            sm_counts=[(128, 0), (96, 32), (80, 48), (64, 64), (0, 128)],
            _pp_prefills={},
            waiting_queue=[request(1, 1000)],
            running_mbs=[SimpleNamespace(reqs=[request(2)])],
            _pp_stream_idx=1,
            _pp_decode_steps=10,
            _pp_required=100,
            _pp_waiting_tokens=0,
            max_prefill_tokens=16384,
            pp_size=4,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=SimpleNamespace(start_layer=0, end_layer=8)
                )
            ),
        )
        p = ALPMuxPPPolicy(s, StaticMuxPPPolicy())
        return p, s

    def test_warmup_dwell_and_hysteresis(self):
        p, s = self.policy()
        cfg = SimpleNamespace(manual_divisions=[[96, 32, 1], [80, 48, 8], [64, 64, 16]])
        self.assertEqual(p.partition(cfg, 5, 1, True), 1)
        p.decode.count = p.prefill.count = 16
        p.decode.theta[2] = 0.01
        p.prefill.theta[4] = 0.001
        self.assertEqual(p.partition(cfg, 5, 1, True), 1)
        s._pp_decode_steps = 20
        self.assertEqual(p.partition(cfg, 5, 1, True), 1)
        self.assertEqual(p.partition(cfg, 5, 1, True), 1)
        self.assertEqual(p.partition(cfg, 5, 1, True), 3)
        self.assertEqual(p.partition(cfg, 5, 1, False), 4)

    def test_admission_uses_model_pressure_and_independent_switch(self):
        p, s = self.policy()
        self.assertEqual(p.admission_budget(s), 16384)
        p.prefill.count = 16
        p.prefill.theta[0] = 0.1
        self.assertEqual(p.admission_budget(s), 1024)
        s._pp_waiting_tokens = 16384
        self.assertEqual(p.admission_budget(s), 1536)
        s.server_args.pdmux_pp_alp_admission = False
        self.assertEqual(p.admission_budget(s), 16384)


class TestAdmission(unittest.TestCase):
    def test_migration_accounts_for_requests_already_assigned_to_target(self):
        s = SchedulerMultiplexPPMixin()
        s.server_args = SimpleNamespace(
            pdmux_pp_rebalance=True,
            pdmux_pp_load_bs_weight=1,
            pdmux_pp_load_ctx_weight=0,
            pdmux_pp_rebalance_unit=1,
            pdmux_pp_rebalance_interval=8,
            max_micro_batch_size=32,
        )
        s.running_mbs = [
            SimpleNamespace(reqs=[request(i) for i in range(10)]),
            SimpleNamespace(reqs=[]),
        ]
        s._pp_pending_merges = [
            [],
            [SimpleNamespace(reqs=[request(i + 10) for i in range(31)])],
        ]
        s._pp_hold = [0, 1]
        packet = SimpleNamespace(step=1, migrations=[], hold_mbs=[])
        s._pp_rebalance_control(packet, [None, None])
        self.assertEqual(packet.migrations, [])
        self.assertEqual(packet.hold_mbs, [])

    def test_reserves_unmerged_output_and_allows_one_request_over_budget(self):
        s = SchedulerMultiplexPPMixin()
        s.server_args = SimpleNamespace(
            pdmux_pp_reserve_decode=True,
            pdmux_pp_balance_merge=True,
            max_micro_batch_size=32,
        )
        s.pp_size, s.max_running_requests, s._pp_max_prefills = 4, 128, 4
        s._pp_admission_budget, s._pp_batch_id = 1024, 1
        s.mux_pp_policy = StaticMuxPPPolicy()
        s.running_mbs = [SimpleNamespace(reqs=[]) for _ in range(4)]
        old = request(0)
        old.sampling_params.max_new_tokens = 5000
        s._pp_prefills = {0: SimpleNamespace(batch=SimpleNamespace(reqs=[old]))}
        s._pp_pending_merges = [[], [], [], []]
        new = request(1, 2000)
        new.sampling_params.max_new_tokens = 1000
        new.fill_ids, new.prefix_indices, new.extend_input_len = [1] * 2000, [], 2000
        new.last_node = None
        new.init_next_round_input = lambda cache: None
        s.waiting_queue = [new]
        s.tree_cache = SimpleNamespace(
            inc_lock_ref=lambda n: None,
            dec_lock_ref=lambda n: None,
            evictable_size=lambda: 0,
        )
        s.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 6000)
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 127)
        from sglang.srt.multiplex.pp_admission import AdmissionStats

        s.max_total_num_tokens = 6000
        s._pp_prefills[0].batch.extend_num_tokens = 2000
        s._pp_admission_stats = AdmissionStats(s)
        admitted = []
        s._pp_build_prefill = lambda plan, reqs: admitted.extend(reqs)
        self.assertEqual(s._pp_make_plan().rids, [])
        s.server_args.pdmux_pp_reserve_decode = False
        self.assertEqual(s._pp_make_plan().rids, [new.rid])
        self.assertEqual(admitted, [new])


class TestBatchSplit(unittest.TestCase):
    def batch(self):
        global_server_args_dict.update(
            enable_deterministic_inference=False, enable_custom_logit_processor=False
        )
        reqs = [request(i) for i in range(6)]
        batch = ScheduleBatch(
            reqs=reqs,
            device="cpu",
            model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=32),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(6) + 30,
            seq_lens=torch.arange(6) + 100,
            orig_seq_lens=torch.arange(6, dtype=torch.int32) + 95,
            seq_lens_cpu_cache=torch.arange(6) + 100,
            output_ids=torch.arange(6) + 1,
            seq_lens_sum=615,
            has_stream=True,
        )
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, 32)
        batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            batch.output_ids
        )
        return batch

    def test_split_merge_preserves_tokens_lengths_and_penalty_history(self):
        batch = self.batch()
        before = torch.zeros(6, 32)
        batch.sampling_info.penalizer_orchestrator.apply(before)
        a, b = batch.split_batch([4, 1, 3])
        self.assertEqual([r.rid for r in a.reqs], ["4", "1", "3"])
        self.assertEqual(a.output_ids.tolist(), [5, 2, 4])
        self.assertIs(a.sampling_info.penalizer_orchestrator.batch, a)
        a.merge_batch(b)
        permutation = [4, 1, 3, 0, 2, 5]
        for field in (
            "req_pool_indices",
            "seq_lens",
            "seq_lens_cpu_cache",
            "orig_seq_lens",
            "output_ids",
        ):
            torch.testing.assert_close(
                getattr(a, field), getattr(batch, field)[permutation]
            )
        after = torch.zeros(6, 32)
        a.sampling_info.penalizer_orchestrator.apply(after)
        torch.testing.assert_close(after, before[permutation])
        a.sampling_info.penalizer_orchestrator.cumulate_output_tokens(a.output_ids)
        unchanged = torch.zeros(6, 32)
        batch.sampling_info.penalizer_orchestrator.apply(unchanged)
        torch.testing.assert_close(before, unchanged)
        self.assertEqual(a.seq_lens_sum, batch.seq_lens_sum)

    def test_full_empty_and_illegal_split(self):
        batch = self.batch()
        a, b = batch.split_batch(list(range(6)))
        self.assertIsNone(b)
        a.output_ids[0] = 31
        self.assertEqual(batch.output_ids[0], 1)
        with self.assertRaises(AssertionError):
            batch.split_batch([0, 0])
        batch.output_ids = None
        with self.assertRaises(AssertionError):
            batch.split_batch([0])


if __name__ == "__main__":
    unittest.main()
