"""Token-row chunking must preserve the SwiGLU map and skip ordinary decode."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.llama import LlamaMLP


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
