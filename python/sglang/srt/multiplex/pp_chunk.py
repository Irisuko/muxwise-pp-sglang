"""Stage-zero chunk plans and allocation-free views of admitted prefill KV.

The parent owns request slots, radix locks and the complete KV allocation until
MERGE. Children only describe attention over successive prefixes of that KV.
"""

import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillChunk:
    # (parent request index, offset in uncached input, number of input tokens)
    spans: tuple[tuple[int, int, int], ...]

    @property
    def tokens(self):
        return sum(n for _, _, n in self.spans)


def plan_chunks(lengths, budget):
    """Pack FCFS input into bounded chunks; no request has two spans per chunk."""
    assert budget > 0 and all(n > 0 for n in lengths)
    chunks, spans, used = [], [], 0
    for index, length in enumerate(lengths):
        offset = 0
        while offset < length:
            n = min(length - offset, budget - used)
            spans.append((index, offset, n))
            used += n
            offset += n
            if used == budget:
                chunks.append(PrefillChunk(tuple(spans)))
                spans, used = [], 0
    if spans:
        chunks.append(PrefillChunk(tuple(spans)))
    return chunks


def chunk_batch(parent, chunk):
    """Slice tensors without allocating/freeing KV or changing shared Req state."""
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    batch = copy.copy(parent)
    first, offset, _ = chunk.spans[0]
    last = chunk.spans[-1][0] + 1
    start = sum(parent.extend_lens[:first]) + offset
    batch.reqs = parent.reqs[first:last]
    batch.input_ids = parent.input_ids[start : start + chunk.tokens]
    batch.out_cache_loc = parent.out_cache_loc[start : start + chunk.tokens]
    batch.req_pool_indices = parent.req_pool_indices[first:last]
    batch.orig_seq_lens = parent.orig_seq_lens[first:last]
    batch.prefix_lens = [parent.prefix_lens[i] + off for i, off, _ in chunk.spans]
    batch.extend_lens = [n for _, _, n in chunk.spans]
    lengths = [p + n for p, n in zip(batch.prefix_lens, batch.extend_lens)]
    batch.seq_lens = torch.tensor(lengths, dtype=torch.int64, device=parent.device)
    batch.pdmux_pp_seq_lens_cpu = torch.tensor(lengths, dtype=torch.int64)
    batch.seq_lens_sum = sum(lengths)
    batch.extend_num_tokens = chunk.tokens
    batch.pdmux_pp_full_prefill_tokens = parent.extend_num_tokens
    batch.pdmux_pp_append_chunk = any(off > 0 for _, off, _ in chunk.spans)
    batch.extend_logprob_start_lens = [0] * len(batch.reqs)
    batch.multimodal_inputs = parent.multimodal_inputs[first:last]
    batch.split_forward_batch = None
    batch.split_prefill_finished = False
    batch.pdmux_pp_final_indices = [
        j
        for j, (i, off, n) in enumerate(chunk.spans)
        if off + n == parent.extend_lens[i]
    ]
    # FCFS packing can leave only the last span unfinished. Terminal logits
    # are therefore a prefix, so sampling can use allocation-free GPU views.
    assert batch.pdmux_pp_final_indices == list(
        range(len(batch.pdmux_pp_final_indices))
    )
    # Only terminal request spans are sampled. Intermediate chunks neither
    # consume RNG nor update penalties, and an all-intermediate chunk skips head.
    final_batch = copy.copy(batch)
    final_batch.reqs = [batch.reqs[i] for i in batch.pdmux_pp_final_indices]
    final_batch.sampling_info = None
    # The penalizer orchestrator keeps a weak reference to its batch. Retain
    # the terminal-request view for the entire chunk's sampling lifetime.
    batch.pdmux_pp_sampling_batch = final_batch if final_batch.reqs else None
    batch.sampling_info = (
        SamplingBatchInfo.from_schedule_batch(
            final_batch, parent.model_config.vocab_size
        )
        if final_batch.reqs
        else None
    )
    return batch


def lane_works(prefills):
    for work in prefills:
        yield from getattr(work, "chunks", None) or (work,)
