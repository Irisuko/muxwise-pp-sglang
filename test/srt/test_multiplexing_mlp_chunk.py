"""Token-row chunking must preserve the SwiGLU map and skip ordinary decode."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.llama import LlamaAttention, LlamaMLP


class LinearPair(nn.Linear):
    def forward(self, x, **kwargs):
        self.rows.append(len(x))
        return super().forward(x), None


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("limit", [1, 4, 7, 23, 100])
@torch.no_grad()
def test_swiglu_math_and_intermediate_bound(limit, pad):
    torch.manual_seed(14)
    mlp = LlamaMLP.__new__(LlamaMLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = LinearPair(8, 24, bias=False, dtype=torch.float64)
    mlp.down_proj = LinearPair(12, 8, bias=False, dtype=torch.float64)
    mlp.gate_up_proj.rows, mlp.down_proj.rows = [], []
    mlp.act_fn = lambda x: torch.nn.functional.silu(x[:, :12]) * x[:, 12:]
    mlp.prefill_chunk_tokens = 0
    batch = SimpleNamespace(forward_mode=ForwardMode.SPLIT_PREFILL)
    x = torch.randn(23, 8, dtype=torch.float64)
    original = x.clone()
    expected = mlp(x, batch)
    mlp.gate_up_proj.rows.clear()
    mlp.prefill_chunk_tokens = limit
    mlp.prefill_pad_to_chunk = pad
    actual = mlp(x, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert max(mlp.gate_up_proj.rows) <= limit
    if pad and len(x) > limit:
        assert mlp.gate_up_proj.rows == [limit] * ((len(x) + limit - 1) // limit)
    else:
        assert sum(mlp.gate_up_proj.rows) == len(x)
    mlp.gate_up_proj.rows.clear()
    batch.forward_mode = ForwardMode.DECODE
    torch.testing.assert_close(mlp(x, batch), expected, rtol=0, atol=0)
    assert mlp.gate_up_proj.rows == [len(x)]


@pytest.mark.parametrize("compact", [False, True])
@torch.no_grad()
def test_compact_mlp_releases_gate_up_before_down_projection(compact):
    mlp = LlamaMLP.__new__(LlamaMLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = LinearPair(8, 24, bias=False, dtype=torch.float64)
    mlp.down_proj = LinearPair(12, 8, bias=False, dtype=torch.float64)
    mlp.gate_up_proj.rows, mlp.down_proj.rows = [], []
    mlp.prefill_pad_to_chunk = compact
    gate_up_ref = None

    def activate(x):
        nonlocal gate_up_ref
        gate_up_ref = weakref.ref(x)
        return torch.nn.functional.silu(x[:, :12]) * x[:, 12:]

    def check_lifetime(module, args):
        assert (gate_up_ref() is None) == compact

    mlp.act_fn = activate
    mlp.down_proj.register_forward_pre_hook(check_lifetime)
    mlp._forward_mlp(torch.randn(4, 8, dtype=torch.float64))


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("full_tokens", [0, 3, 19])
@pytest.mark.parametrize("mode", [ForwardMode.SPLIT_PREFILL, ForwardMode.DECODE])
@torch.no_grad()
def test_pipeline_tail_retains_full_prefill_mlp_padding(pad, full_tokens, mode):
    mlp = LlamaMLP.__new__(LlamaMLP)
    nn.Module.__init__(mlp)
    mlp.prefill_chunk_tokens = 8
    mlp.prefill_pad_to_chunk = pad
    rows = []

    def mlp_rows(x, _):
        rows.append(len(x))
        if len(x) > 3:
            assert not torch.count_nonzero(x[3:])
        return x * 2

    mlp._forward_mlp = mlp_rows
    x = torch.randn(3, 5)
    batch = SimpleNamespace(forward_mode=mode, pdmux_pp_full_prefill_tokens=full_tokens)
    torch.testing.assert_close(mlp(x, batch), 2 * x, rtol=0, atol=0)
    padded = pad and full_tokens > 8 and mode == ForwardMode.SPLIT_PREFILL
    assert rows == [8 if padded else 3]


@pytest.mark.parametrize("full_tokens", [0, 3, 19])
@pytest.mark.parametrize(
    "mode", [ForwardMode.SPLIT_PREFILL, ForwardMode.EXTEND, ForwardMode.DECODE]
)
@torch.no_grad()
def test_projection_padding_never_adds_attention_queries(full_tokens, mode):
    layer = LlamaAttention.__new__(LlamaAttention)
    nn.Module.__init__(layer)
    layer.qkv_proj = LinearPair(5, 8, bias=False, dtype=torch.float64)
    layer.o_proj = LinearPair(4, 5, bias=False, dtype=torch.float64)
    layer.qkv_proj.rows, layer.o_proj.rows = [], []
    layer.q_size, layer.kv_size = 4, 2

    def rope(positions, q, k):
        assert positions.tolist() == [16, 17, 18]
        assert len(q) == len(k) == 3
        return q, k

    def attention(q, k, v, _):
        assert len(q) == len(k) == len(v) == 3
        return q + torch.cat([k + v, k + v], dim=1)

    layer.rotary_emb, layer.attn = rope, attention
    batch = SimpleNamespace(forward_mode=mode, pdmux_pp_full_prefill_tokens=full_tokens)
    inputs, positions = torch.randn(3, 5, dtype=torch.float64), torch.arange(16, 19)
    expected = layer(positions, inputs, batch)
    layer.qkv_proj.rows.clear()
    layer.o_proj.rows.clear()
    layer.prefill_projection_pad_to = 8
    actual = layer(positions, inputs, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    padded = full_tokens > 8 and mode == ForwardMode.SPLIT_PREFILL
    assert layer.qkv_proj.rows == layer.o_proj.rows == [8 if padded else 3]


@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize("mode", [ForwardMode.SPLIT_PREFILL, ForwardMode.DECODE])
@torch.no_grad()
def test_cpp_padding_preserves_gemm_but_excludes_dummy_tp_rows(
    monkeypatch, reduce_results, mode
):
    """Two identical shards must sum once, communicating only real token rows."""
    reduced_shapes = []

    def reduce(tensor):
        reduced_shapes.append(tuple(tensor.shape))
        assert tensor.is_contiguous()
        return tensor * 2

    class TPLinearPair(LinearPair):
        tp_size = 2

        def forward(self, x, skip_all_reduce=False):
            output, bias = super().forward(x)
            if self.reduce_results and not skip_all_reduce:
                output = reduce(output)
            return output, bias

    monkeypatch.setattr(
        "sglang.srt.models.llama.tensor_model_parallel_all_reduce", reduce
    )
    torch.manual_seed(12)
    layer = LlamaAttention.__new__(LlamaAttention)
    nn.Module.__init__(layer)
    layer.q_size, layer.kv_size = 4, 2
    layer.qkv_proj = LinearPair(5, 8, bias=False, dtype=torch.float64)
    layer.o_proj = TPLinearPair(4, 5, bias=False, dtype=torch.float64)
    layer.o_proj.reduce_results = reduce_results
    layer.qkv_proj.rows, layer.o_proj.rows = [], []
    layer.rotary_emb = lambda positions, q, k: (q, k)
    layer.attn = lambda q, k, v, batch: q + torch.cat([k + v, k + v], dim=1)
    batch = SimpleNamespace(forward_mode=mode, pdmux_pp_full_prefill_tokens=19)
    inputs, positions = torch.randn(3, 5, dtype=torch.float64), torch.arange(3)
    expected = layer(positions, inputs, batch)
    reduced_shapes.clear()
    layer.qkv_proj.rows.clear()
    layer.o_proj.rows.clear()
    layer.prefill_projection_pad_to = 8
    actual = layer(positions, inputs, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert reduced_shapes == ([(3, 5)] if reduce_results else [])
    gemm_rows = 8 if mode == ForwardMode.SPLIT_PREFILL else 3
    assert layer.qkv_proj.rows == layer.o_proj.rows == [gemm_rows]
