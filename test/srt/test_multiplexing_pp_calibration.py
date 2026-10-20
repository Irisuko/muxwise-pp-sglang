"""Safety gates must not extrapolate to unmeasured SM/context combinations."""

import math
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multiplex.pp_policy import calibrated_decode_time, choose_partition
from sglang.srt.multiplex.pp_timing import LaneTimings


def test_calibration_uses_upper_corner_and_monotonic_envelope():
    rows = [(1, 2000, 0.003), (1, 8000, 0.005), (8, 2000, 0.004), (8, 8000, 0.0045)]
    assert calibrated_decode_time(rows, 4, 3000) == 0.005
    assert math.isinf(calibrated_decode_time(rows, 9, 3000))
    assert math.isinf(calibrated_decode_time(rows, 4, 32000))
    assert math.isinf(calibrated_decode_time([], 1, 1))


def test_infinite_demand_chooses_largest_safe_prefill_even_with_infinite_rate():
    sm = [
        (128, 0),
        (120, 8),
        (112, 16),
        (104, 24),
        (96, 32),
        (80, 48),
        (64, 64),
        (0, 128),
    ]
    td = [0.001, 0.02, 0.01, 0.008, 0.006, 0.004, 0.003, 0.002]
    rates = [float("inf")] * len(sm)
    assert choose_partition(sm, td, rates, float("inf"), 4, 0.1) == 1
    td[1] = float("nan")
    assert choose_partition(sm, td, rates, float("inf"), 4, 0.1) == 2
    td[2] = 0.026
    assert choose_partition(sm, td, rates, float("inf"), 4, 0.1) == 3


def test_lane_wall_accounting_partitions_and_wait_phases():
    clock = [0.0]
    s = SimpleNamespace(_pp_prefills={})
    with patch(
        "sglang.srt.multiplex.pp_timing.time.monotonic", side_effect=lambda: clock[0]
    ):
        timer = LaneTimings(s)
        clock[0] = 2
        s._pp_prefills[0] = SimpleNamespace(state="waiting")
        timer.prefill_state()
        clock[0] = 5
        s._pp_prefills[0].state = "offering"
        timer.prefill_state()
        timer.switch(1)
        clock[0] = 7
        with timer.region("prefill", "switch"):
            clock[0] = 8
        clock[0] = 10
        result = timer.snapshot()
    assert result["wall"]["prefill"] == {
        "empty": 2,
        "upstream": 3,
        "ready": 4,
        "switch": 1,
    }
    assert result["partitions"] == {0: 5, 1: 5}
    assert sum(result["wall"]["decode"].values()) == 10


def test_prefill_fit_is_invariant_to_slice_granularity_and_final_head_cost():
    import numpy as np

    from sglang.srt.multiplex.pp_policy import ALPMuxPPPolicy

    scheduler = SimpleNamespace(server_args=None, sm_counts=[(96, 32)])
    split = ALPMuxPPPolicy(scheduler, None)
    whole = ALPMuxPPPolicy(scheduler, None)
    for batch in range(40):
        base = [batch % 8 + 1, (batch % 8 + 1) ** 2, batch % 4, batch % 3 + 1, 1]
        body = 0.01 * base[0]
        head = 0.005 * base[3]
        for layer in range(8):
            split.observe_prefill_slice(
                batch, base, 0, body + (head if layer == 7 else 0), 1, layer == 7
            )
        whole.observe_prefill_slice(batch, base, 0, 8 * body + head, 8, True)
    assert split.prefill.count == whole.prefill.count == 40
    assert not split.prefill_pending
    np.testing.assert_allclose(
        split.prefill.theta, whole.prefill.theta, rtol=1e-10, atol=1e-12
    )
    assert split.samples_by_partition[0][1] == 320
    assert whole.samples_by_partition[0][1] == 40


def test_profile_ragged_contexts_stay_in_pool_bounds_and_preserve_work():
    from sglang.srt.multiplex.pp_profile import context_shapes

    for bs in (1, 4, 8, 16, 32):
        for context in (2048, 4096, 8192, 16384, 33792):
            for shape in context_shapes(bs, context, 33792):
                assert len(shape) == bs
                assert sum(shape) == bs * context
                assert min(shape) >= 1 and max(shape) <= 33792


def test_endpoint_transition_counts_gpu_time_without_poisoning_rls():
    from unittest.mock import Mock

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 2.0

    models = Mock()
    scheduler = SimpleNamespace(
        sm_counts=[(128, 0)], _pp_stream_idx=0, _pp_models=models
    )
    timer = LaneTimings(scheduler)
    with patch("sglang.srt.multiplex.pp_timing.torch.cuda.Event", Event):
        timer.end(timer.begin("decode", [1, 1, 1]))
        timer.drain()
    assert timer.cuda["decode"] == 0.002
    models.observe.assert_not_called()


def test_rls_recovers_from_negative_linear_prediction_instead_of_floor_lock():
    from sglang.srt.multiplex.pp_policy import OnlineRLS

    model = OnlineRLS(3, 2)
    model.count = 16
    model.theta[2] = -0.01
    model.update([1, 1, 1], 0.02)
    assert model.predict([1, 1, 1]) > 0.01


def test_rls_cold_fit_learns_workload_coefficient_without_large_bias_seed():
    from sglang.srt.multiplex.pp_policy import OnlineRLS

    model = OnlineRLS(3, 2)
    model.update([32, 1024, 1], 0.2)
    assert model.theta[2] < 0.001
    assert abs(model.predict([32, 1024, 1]) - 0.2) < 0.001


def test_unsafe_current_partition_bypasses_hysteresis_and_minimum_residency():
    from sglang.srt.multiplex.pp_policy import ALPMuxPPPolicy

    args = SimpleNamespace(
        pdmux_pp_slo_tpot=0.1,
        pdmux_pp_tpot_coeff=0.75,
        pdmux_pp_min_residency_steps=100,
        pdmux_pp_hysteresis_steps=100,
    )
    req = SimpleNamespace(origin_input_ids=list(range(100)), output_ids=[])
    scheduler = SimpleNamespace(
        server_args=args,
        sm_counts=[(128, 0), (120, 8), (96, 32), (0, 128)],
        running_mbs=[SimpleNamespace(reqs=[req])],
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(start_layer=0, end_layer=8)
            )
        ),
        pp_size=4,
        _pp_stream_idx=1,
        _pp_decode_steps=1,
        _pp_required=float("inf"),
    )
    policy = ALPMuxPPPolicy(scheduler, SimpleNamespace(partition=lambda *args: 1))
    policy._prefill_base = lambda: [1, 1, 0, 1, 1]
    policy.decode_calibration = {1: [(1, 2048, 0.03)], 2: [(1, 2048, 0.005)]}
    assert policy.partition(None, 4, 1, True) == 2
