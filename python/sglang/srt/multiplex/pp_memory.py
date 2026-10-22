"""Opt-in developer memory measurements; no synchronization when disabled."""

import json
import os

import torch

PREFIX = os.getenv("SGLANG_PDMUX_PP_MEMORY_TRACE")


def record(event, **extra):
    if not PREFIX:
        return
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    row = dict(
        event=event,
        pid=os.getpid(),
        device=torch.cuda.current_device(),
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak=torch.cuda.max_memory_allocated(),
        free=free,
        total=total,
        inactive_split=stats.get("inactive_split_bytes.all.current", 0),
        **extra,
    )
    with open(f"{PREFIX}-{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def install(runner):
    if not PREFIX:
        return
    params = {}
    for name, p in runner.model.named_parameters():
        key = "layers" if ".layers." in name else name
        params[key] = params.get(key, 0) + p.numel() * p.element_size()
    workspaces = {}
    seen = set()

    def tensors(value, name, depth=0):
        if id(value) in seen or depth > 4:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                workspaces[name] = dict(
                    bytes=value.numel() * value.element_size(), ptr=value.data_ptr()
                )
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                tensors(item, f"{name}.{i}", depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                tensors(item, f"{name}.{key}", depth + 1)
        elif hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
            for key, item in vars(value).items():
                if key not in (
                    "model_runner",
                    "req_to_token",
                    "req_to_token_pool",
                    "token_to_kv_pool",
                ):
                    tensors(item, f"{name}.{key}", depth + 1)

    for i, backend in enumerate(
        [runner.attn_backend] + getattr(runner, "decode_attn_backend_group", [])
    ):
        tensors(backend, str(i))
    pools = {}
    for segment in torch.cuda.memory_snapshot():
        key = str(segment.get("segment_pool_id", "unknown"))
        pool = pools.setdefault(key, dict(reserved=0, allocated=0, active=0))
        for src, dst in [
            ("total_size", "reserved"),
            ("allocated_size", "allocated"),
            ("active_size", "active"),
        ]:
            pool[dst] += segment[src]
    record(
        "ready",
        pp=runner.pp_rank,
        tp=runner.tp_rank,
        params=params,
        backend_tensors=workspaces,
        graph_pools=pools,
    )
    model = runner.model.model
    for index in range(model.start_layer, model.end_layer):
        layer = model.layers[index]
        for name, module in [
            ("layer", layer),
            ("attention", layer.self_attn),
            ("mlp", layer.mlp),
        ]:

            def before(mod, args, kwargs, idx=index, kind=name):
                tensors = [
                    x
                    for x in (*args, *kwargs.values())
                    if isinstance(x, torch.Tensor) and x.ndim >= 1
                ]
                mod._p10_profile = bool(
                    tensors and max(x.shape[0] for x in tensors) >= 30000
                )
                if mod._p10_profile:
                    torch.cuda.current_stream().synchronize()
                    if kind != "layer":
                        torch.cuda.reset_peak_memory_stats()
                    record("before_" + kind, layer=idx)

            def after(mod, args, kwargs, output, idx=index, kind=name):
                if getattr(mod, "_p10_profile", False):
                    torch.cuda.current_stream().synchronize()
                    record("after_" + kind, layer=idx)

            module.register_forward_pre_hook(before, with_kwargs=True)
            module.register_forward_hook(after, with_kwargs=True)
