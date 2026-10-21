"""Bounded startup calibration of decode graphs, before any request owns KV.

The table measures each partition directly, including long-context attention;
SM extrapolation is deliberately not used as a safety gate. No graph is captured
here and no request/cache/allocator state is allocated or published.
"""

import logging
import time

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


def context_shapes(bs, mean_context, max_context):
    uniform = [mean_context] * bs
    skewed = [1] * bs
    remaining = bs * (mean_context - 1)
    for i in range(bs):
        extra = min(remaining, max_context - 1)
        skewed[i] += extra
        remaining -= extra
    assert remaining == 0
    return [uniform] if uniform == skewed else [uniform, skewed]


def profile_decode(scheduler, max_seconds=55):
    runner = scheduler.tp_worker.model_runner
    graph = runner.graph_runner
    if graph is None or not hasattr(runner, "decode_attn_backend_group"):
        raise ValueError("PP decode calibration requires partition CUDA graphs")
    started = time.monotonic()
    table = {}
    # Pad to captured sizes at runtime as well. Cover the entire configured
    # context range, so a 31k request does not depend on SM/context extrapolation.
    sizes = sorted(
        set(
            [
                graph.capture_bs[0],
                *[b for b in graph.capture_bs if b in (4, 8, 16)],
                graph.max_bs,
            ]
        )
    )
    contexts = sorted(
        set(
            min(c, scheduler.model_config.context_len)
            for c in (2048, 4096, 8192, 16384, 33792)
        )
    )
    mapping = runner.req_to_token_pool.req_to_token
    width = max(contexts)
    saved_mapping = mapping[: graph.max_bs, :width].clone()
    saved = {
        name: getattr(graph, name).clone()
        for name in (
            "seq_lens",
            "req_pool_indices",
            "out_cache_loc",
            "positions",
            "num_token_non_padded",
        )
    }
    try:
        # Distinct physical pages, wrapping only if bs*context exceeds the pool.
        # All are unowned at startup; allocation and radix metadata stay untouched.
        capacity = scheduler.max_total_num_tokens
        mapping[: graph.max_bs, :width] = (
            torch.arange(graph.max_bs * width, device=mapping.device).reshape(
                graph.max_bs, width
            )
            % (capacity - 1)
        ) + 1
        graph.req_pool_indices.copy_(
            torch.arange(
                graph.max_bs, device=mapping.device, dtype=graph.req_pool_indices.dtype
            )
        )
        graph.out_cache_loc[: graph.max_bs].copy_(mapping[: graph.max_bs, 0])
        torch.cuda.synchronize()
        for idx, (_, decode_sm) in enumerate(scheduler.sm_counts):
            if not decode_sm:
                continue
            stream = scheduler.stream_groups[idx][1]
            backend = runner.decode_attn_backend_group[idx]
            records = []
            for bs in sizes:
                for context in contexts:
                    if not scheduler._pp_tp_all(
                        time.monotonic() - started <= max_seconds
                    ):
                        raise RuntimeError(
                            "PP decode calibration exceeded startup time limit"
                        )
                    seconds = 0.0
                    for shape in context_shapes(bs, context, width):
                        if scheduler.tp_size > 1:
                            dist.barrier(group=scheduler.tp_cpu_group)
                        with torch.cuda.stream(stream):
                            graph.num_token_non_padded.fill_(bs)
                            cpu_lens = torch.tensor(shape, dtype=torch.int32)
                            graph.seq_lens[:bs].copy_(cpu_lens)
                            graph.positions[:bs].copy_(cpu_lens - 1)
                            backend.init_forward_metadata_replay_cuda_graph(
                                bs,
                                graph.req_pool_indices[:bs],
                                graph.seq_lens[:bs],
                                bs * context,
                                None,
                                ForwardMode.DECODE,
                                None,
                                seq_lens_cpu=cpu_lens,
                            )
                            replay = graph.graphs[f"{idx}_{bs}"].replay
                            replay()
                            start, end = torch.cuda.Event(
                                enable_timing=True
                            ), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(3):
                                replay()
                            end.record()
                        end.synchronize()
                        seconds = max(seconds, start.elapsed_time(end) / 3000)
                    records.append((bs, context, seconds))
            table[idx] = records
            logger.info(
                "MuxWise-PP decode calibration partition=%s SMs=%s samples=%s",
                idx,
                scheduler.sm_counts[idx],
                records,
            )
    finally:
        torch.cuda.synchronize()
        mapping[: graph.max_bs, :width].copy_(saved_mapping)
        for name, value in saved.items():
            getattr(graph, name).copy_(value)
        torch.cuda.synchronize()
    logger.info(
        "MuxWise-PP decode calibration elapsed=%.3fs", time.monotonic() - started
    )
    return table
