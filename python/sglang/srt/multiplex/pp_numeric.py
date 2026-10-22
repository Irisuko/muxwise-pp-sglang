"""Opt-in, single-request layer dumps for developer numerical comparisons."""

import os

import torch

PREFIX = os.getenv("SGLANG_PDMUX_PP_NUMERIC_TRACE")


def install(runner):
    if not PREFIX:
        return
    prefix = f"{PREFIX}-pp{runner.pp_rank}-tp{runner.tp_rank}"
    state = dict(active=False, step=-1)
    model = runner.model.model

    def pack(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2:
                n = value.shape[0]
                ix = sorted(
                    set(
                        [0, n - 1]
                        + [
                            x
                            for x in (1, 100, 1023, 1024, 2047, 2048, 4095, 4096)
                            if x < n
                        ]
                        + list(range(max(0, n - 32), n))
                    )
                )
                return dict(
                    shape=tuple(value.shape),
                    rows=ix,
                    value=value[ix].detach().float().cpu(),
                )
            return value.detach().cpu()
        if isinstance(value, (tuple, list)):
            return [pack(x) for x in value]
        return None

    def start(module, args, kwargs):
        tensors = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]
        n = max((v.shape[0] for v in tensors if v.ndim), default=0)
        if n >= 1000:
            state["active"] = True
            state["step"] = 0
        elif state["active"]:
            state["step"] += 1

    model.layers[model.start_layer].register_forward_pre_hook(start, with_kwargs=True)
    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        for kind, module in [
            ("layer", layer),
            ("mlp", layer.mlp),
            ("attention", layer.self_attn),
        ]:

            def dump(mod, args, kwargs, result, index=i, name=kind):
                if state["active"] and state["step"] <= 2:
                    torch.save(
                        dict(
                            input=pack(
                                [
                                    x
                                    for x in (*args, *kwargs.values())
                                    if isinstance(x, torch.Tensor)
                                ]
                            ),
                            output=pack(result),
                        ),
                        f'{prefix}-step{state["step"]}-{index}-{name}.pt',
                    )

            module.register_forward_hook(dump, with_kwargs=True)

    def logits(mod, args, result):
        if state["active"] and state["step"] <= 2:
            torch.save(
                result.next_token_logits.detach().float().cpu(),
                f'{prefix}-step{state["step"]}-logits.pt',
            )

    runner.model.logits_processor.register_forward_hook(logits)

    graph = runner.graph_runner
    if graph is not None:
        original_replay = graph.replay

        def replay(*args, **kwargs):
            result = original_replay(*args, **kwargs)
            if state["active"]:
                state["step"] += 1
                if state["step"] <= 2:
                    if hasattr(result, "next_token_logits"):
                        torch.save(
                            result.next_token_logits.detach().float().cpu(),
                            f'{prefix}-step{state["step"]}-logits.pt',
                        )
                    elif hasattr(result, "tensors"):
                        torch.save(
                            {k: pack(v) for k, v in result.tensors.items()},
                            f'{prefix}-step{state["step"]}-proxy.pt',
                        )
            return result

        graph.replay = replay
