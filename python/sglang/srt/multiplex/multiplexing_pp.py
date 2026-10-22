"""MuxWise PP: ordered decode controls and an independent asynchronous prefill lane.

Only the scheduler thread touches request/cache state. Daemon threads
transport CPU envelopes/readiness; another launches NCCL on the selected prefill stream
(even ncclGroupEnd can block the host on first peer connection). In particular, an
unmatched receive is never followed by a stream synchronize/partition change.
"""

import logging
import math
import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    get_pdmux_prefill_pp_group,
    set_pdmux_status,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.evict_policy import LRUStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.multiplex.pdmux_context import set_current_stream_idx
from sglang.srt.multiplex.pp_admission import AdmissionStats
from sglang.srt.multiplex.pp_chunk import (
    PrefillChunk,
    chunk_batch,
    lane_works,
    plan_chunks,
)
from sglang.srt.multiplex.pp_policy import (
    ALPMuxPPPolicy,
    balanced_assignment,
    local_prefill_runnable,
    migration_rids,
    rebalance_pair,
    req_context,
    required_throughput,
    work_bounded_inflight_limit,
)
from sglang.srt.multiplex.pp_timing import LaneTimings

logger = logging.getLogger(__name__)


class ControlKind(str, Enum):
    ADMIT = "admit"
    MERGE = "merge"
    RETRACT = "retract"
    ABORT = "abort"
    MIGRATE = "migrate"


@dataclass
class PrefillPlan:
    batch_id: int
    # Include unsuccessful probes: match_prefix can split radix nodes.
    probes: list[tuple[str, int, int]]  # rid, fill length, prefix length
    rids: list[str]
    chunks: list[PrefillChunk] = field(default_factory=list)


@dataclass
class MigrationPlan:
    """P2 command, applied at a control step where both microbatches are retired."""

    rids: list[str]
    source_mb: int
    target_mb: int


@dataclass
class PPControl:
    step: int
    mb_id: int
    requests: list = field(default_factory=list)
    plan: Optional[PrefillPlan] = None
    merge: Optional[dict] = None  # batch_id, target_mb, next_token_ids
    retract: list[str] = field(default_factory=list)
    abort: list[str] = field(default_factory=list)
    migrations: list[MigrationPlan] = field(default_factory=list)
    evictions: list = field(default_factory=list)
    hold_mbs: list[int] = field(default_factory=list)
    required: float = 0.0
    waiting_tokens: int = 0
    admission_budget: int = 0
    decode_bs: int = 0
    retired: Optional[dict] = None
    # Telemetry only: each upstream stage stamps its own forecast before relay.
    decode_notice: Optional[dict] = None


class StaticMuxPPPolicy:
    """Static fallback/ablation; admission and merge hooks run on stage zero."""

    def partition(self, config, group_count, decode_bs, has_prefill):
        if not decode_bs:
            return 0
        if not has_prefill:
            return group_count - 1
        idx = 1
        if config.manual_divisions:
            for i, (_, _, threshold) in enumerate(config.manual_divisions):
                if decode_bs >= threshold:
                    idx = i + 1
            return idx
        return max(
            1,
            min(
                group_count - 2,
                decode_bs * (group_count - 2) // config.decode_bs_divisor,
            ),
        )

    def admission_budget(self, scheduler):
        return scheduler.max_prefill_tokens

    def merge_target(self, running_mbs):
        return min(
            range(len(running_mbs)),
            key=lambda i: (sum(not r.finished() for r in running_mbs[i].reqs), i),
        )

    def slice_layers(self, config, token_count, remaining_layers):
        return min(
            remaining_layers,
            max(1, config.split_forward_token_budget // max(1, token_count)),
        )


class DeterministicRadixFIFO:
    """Node creation order is identical under the ordered control protocol.

    Wall-clock LRU is unsafe when stages progress at different rates. FIFO keeps
    radix prefix reuse, while avoiding any dependence on local timestamps.
    """

    def get_priority(self, node):
        return node.id


@dataclass
class PrefillWork:
    plan: PrefillPlan
    batch: ScheduleBatch
    state: str
    event: object = None
    launched: object = None
    proxy: object = None
    tokens: object = None
    recv_slot: Optional[int] = None
    trace_started: bool = False
    trace_slice: bool = False
    chunks: list = field(default_factory=list)
    parent_id: Optional[int] = None
    chunk_index: int = 0
    completed_tokens: int = 0
    acked_chunks: int = 0
    sampled_tokens: dict = field(default_factory=dict)


class PrefillReceivePool:
    """Persistent input slots; an optional second slot receives ahead.

    Residuals can alias received storage throughout the layer stack, so a slot
    remains owned until the final compute/send event has completed. Preallocation
    also avoids cudaMalloc synchronizing an unmatched NCCL receive at runtime.
    """

    def __init__(self, max_tokens, hidden_size, dtype, device, slots=2):
        assert slots in (1, 2)
        self.buffers = [
            {
                name: torch.empty((max_tokens, hidden_size), dtype=dtype, device=device)
                for name in ("hidden_states", "residual")
            }
            for _ in range(slots)
        ]
        self.free = deque(range(slots))

    def acquire(self, specs):
        assert {name for name, _, _ in specs} == set(self.buffers[0])
        for name, shape, dtype in specs:
            storage = self.buffers[0][name]
            assert len(shape) == 2 and 0 < shape[0] <= storage.shape[0]
            assert shape[1] == storage.shape[1] and dtype == storage.dtype
        if not self.free:
            return None
        slot = self.free.popleft()
        return slot, {
            name: self.buffers[slot][name][: shape[0]] for name, shape, _ in specs
        }

    def assert_idle(self):
        assert sorted(self.free) == list(range(len(self.buffers)))

    def release(self, slot):
        assert 0 <= slot < len(self.buffers) and slot not in self.free
        self.free.append(slot)


class OrderedRadixEviction:
    """Journal each allocator eviction; followers replay exact full token paths.

    Admission and decode allocation run in control order. Cache insertion only
    happens at MERGE/result retirement, so unmerged prefill is never matchable.
    A path includes extra_key and survives rank-specific node IDs/timestamps.
    """

    def __init__(self, cache, leader):
        self.cache, self.leader = cache, leader
        self.original_evict = cache.evict
        self.commands = None
        self.cursor = 0
        cache.evict = self.evict

    def begin(self, commands):
        self.commands, self.cursor = commands, 0

    def end(self):
        if not self.leader:
            assert self.cursor == len(self.commands), "Missing PP eviction replay"
        self.commands = None

    @staticmethod
    def path(node):
        parts, extra = [], node.key.extra_key
        while node.parent is not None:
            parts.append(node.key.token_ids)
            node = node.parent
        return extra, [t for part in reversed(parts) for t in part]

    def evict(self, num_tokens):
        assert self.commands is not None, "Eviction outside PP control step"
        cache = self.cache
        if self.leader:
            victims = []
            delete = cache._delete_leaf

            def record(node):
                victims.append(self.path(node))
                delete(node)

            cache._delete_leaf = record
            try:
                self.original_evict(num_tokens)
            finally:
                cache._delete_leaf = delete
            self.commands.append((num_tokens, victims))
        else:
            expected, victims = self.commands[self.cursor]
            self.cursor += 1
            assert expected == num_tokens, (expected, num_tokens)
            from sglang.srt.mem_cache.radix_cache import RadixKey

            for extra, tokens in victims:
                node = cache.root_node
                remaining = RadixKey(tokens, extra)
                while len(remaining):
                    node = node.children[cache.get_child_key_fn(remaining)]
                    assert remaining.token_ids[: len(node.key)] == node.key.token_ids
                    remaining = remaining[len(node.key) :]
                assert node is not cache.root_node and not node.children
                assert node.lock_ref == 0, "Stage0 selected a protected radix node"
                cache.token_to_kv_pool_allocator.free(node.value)
                cache._delete_leaf(node)
                cache._record_remove_event(node)


class PrefillMailbox:
    """Blocking Gloo I/O lives off the scheduler thread. No CUDA in these threads."""

    def __init__(self, group):
        self.group = group
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        self.errors = queue.Queue()
        self.transfers = queue.Queue()
        self.pending_sends = {}
        self.send_lock = threading.Lock()
        self.ready = set()
        self.src = group.ranks[(group.rank_in_group - 1) % group.world_size]
        self.dst = group.ranks[(group.rank_in_group + 1) % group.world_size]
        for target, name in (
            (self._receive, "rx"),
            (self._send, "tx"),
            (self._transfer, "cuda"),
            (self._receive_ready, "ready"),
        ):
            threading.Thread(
                target=self._guard,
                args=(target,),
                daemon=True,
                name=f"pdmux-prefill-{name}",
            ).start()

    def _guard(self, target):
        try:
            target()
        except Exception as exc:
            self.errors.put(exc)

    def _receive(self):
        while True:
            envelope = [None]
            dist.recv_object_list(envelope, src=self.src, group=self.group.cpu_group)
            self.inbox.put(envelope[0])

    def _receive_ready(self):
        if self.group.rank_in_group == self.group.world_size - 1:
            return
        while True:
            bid = torch.empty(1, dtype=torch.int64)
            dist.recv(bid, src=self.dst, group=self.group.cpu_group, tag=317)
            self._dispatch_ready(int(bid.item()))

    def _dispatch_ready(self, bid):
        with self.send_lock:
            task = self.pending_sends.pop(bid, None)
        if task is None:
            self.inbox.put(dict(kind="ready", batch_id=bid))
        else:
            self.transfers.put(task)

    def send_when_ready(self, bid, tensors, stream):
        # Register BEFORE publishing the hidden envelope. READY can then launch
        # the matching send without another decode/control turn. A scheduler
        # D2H copy must not delay the peer of an already-posted NCCL receive.
        launched, event = threading.Event(), torch.cuda.Event()
        with self.send_lock:
            assert bid not in self.pending_sends
            self.pending_sends[bid] = (True, tensors, stream, launched, event)
        return launched, event

    def _send(self):
        while True:
            envelope = self.outbox.get()
            if envelope is None:
                return
            if envelope["kind"] == "ready":
                dist.send(
                    torch.tensor([envelope["batch_id"]], dtype=torch.int64),
                    dst=self.src,
                    group=self.group.cpu_group,
                    tag=317,
                )
            else:
                dist.send_object_list(
                    [envelope], dst=self.dst, group=self.group.cpu_group
                )

    def poll(self):
        if not self.errors.empty():
            raise RuntimeError("Prefill communication failed") from self.errors.get()
        try:
            return self.inbox.get_nowait()
        except queue.Empty:
            return None

    def _transfer(self):
        with torch.inference_mode(), torch.cuda.device(self.group.device):
            while True:
                task = self.transfers.get()
                if task is None:
                    return
                sending, tensors, stream, launched, event = task
                with torch.cuda.stream(stream):
                    comm = self.group.pynccl_comm
                    comm.group_start()
                    for tensor in tensors.values():
                        if sending:
                            comm.send(tensor, self.group.rank_in_group + 1)
                        else:
                            comm.recv(tensor, self.group.rank_in_group - 1)
                    comm.group_end()
                    event.record(stream)
                launched.set()
                # Do not retain a previous batch while waiting for the next task.
                del task, tensors, tensor

    def transfer(self, sending, tensors, stream):
        launched, event = threading.Event(), torch.cuda.Event()
        self.transfers.put((sending, tensors, stream, launched, event))
        return launched, event


class SchedulerMultiplexPPMixin:
    def _pp_lane_works(self):
        return lane_works(self._pp_prefills.values())

    def _pp_validate_request(self, req, recv_req):
        """Fail unsupported P1 features before allocating pools/grammar state."""
        params = req.sampling_params
        unsupported = (
            req.return_logprob
            or req.return_hidden_states
            or req.input_embeds is not None
            or recv_req.mm_inputs is not None
            or recv_req.session_params is not None
            or any(
                getattr(params, name, None) is not None
                for name in ("json_schema", "regex", "ebnf", "structural_tag")
            )
        )
        if unsupported:
            from sglang.srt.disaggregation.utils import prepare_abort

            prepare_abort(
                req,
                "MuxWise-PP supports text/token generation without "
                "logprobs, hidden-state output, sessions, or constrained decoding.",
                status_code=HTTPStatus.BAD_REQUEST,
            )
            self.stream_output([req], req.return_logprob)
            return False
        return True

    def _pp_result(self, tokens, can_run_cuda_graph=False):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=tokens,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            bid=None,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _pp_init(self):
        # Only this rank may decide request/cache transitions. Stage0 TP peers
        # replay its complete journal before forwarding along their PP pair.
        self._pp_leader = self.pp_rank == 0 and self.tp_rank == 0
        # MistralForCausalLM inherits Llama's PP and split-prefill implementation.
        assert self.model_config.hf_config.model_type in (
            "llama",
            "mistral",
        ), "MuxWise-PP supports Llama and Mistral"
        assert (
            type(self.tree_cache) is RadixCache
        ), "MuxWise-PP requires the Python RadixCache"
        self.tree_cache.eviction_strategy = (
            DeterministicRadixFIFO()
            if self.server_args.pdmux_pp_radix_eviction == "fifo"
            else LRUStrategy()
        )
        self._pp_eviction = OrderedRadixEviction(self.tree_cache, self._pp_leader)
        self._pp_models = ALPMuxPPPolicy(self, StaticMuxPPPolicy())
        self.mux_pp_policy = (
            self._pp_models
            if self.server_args.pdmux_pp_policy == "alp"
            else self._pp_models.fallback
        )
        self._pp_timing = LaneTimings(self)
        self.tp_worker.pdmux_pp_timing = self._pp_timing
        self.tp_worker.model_runner.model.pdmux_pp_timing = self._pp_timing
        self._pp_clock_host = socket.gethostname()
        self._pp_upstream_decode = None
        self._pp_slice_choices = [0, 0, 0]  # planned, shortened, sum(base_count-count)
        self._pp_required = 0.0
        self._pp_waiting_tokens = 0
        self._pp_admission_budget = self.max_prefill_tokens
        self._pp_budget_stats = dict(calls=0, skipped=0, seconds=0.0)
        self._pp_arrivals = deque()
        self._pp_pending_merges = [[] for _ in range(self.pp_size)]
        self._pp_hold = []
        self._pp_migrations = 0
        self._pp_stage_status = {}
        self._pp_last_summary = time.monotonic()
        self.prefill_pp_group = get_pdmux_prefill_pp_group()
        self.prefill_mailbox = PrefillMailbox(self.prefill_pp_group)
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        self._pp_batch_id = 0
        self._pp_prefills = {}  # insertion order = stage0 ADMIT order
        self._pp_envelopes = {}
        self._pp_merge_ready = deque()
        self._pp_max_prefills = (
            self.server_args.pdmux_pp_max_inflight_prefill or self.pp_size
        )
        self._pp_admission_stats = AdmissionStats(self)
        self._pp_stream_idx = 0
        self._pp_idle_logged = False
        self._pp_prefill_steps = 0
        self._pp_decode_steps = 0
        self._pp_cached_tokens = 0
        self._pp_overlap_launches = 0
        self._pp_select_streams(0)
        if self.pdmux_config.pp_profile_decode:
            from sglang.srt.multiplex.pp_profile import profile_decode

            self._pp_models.decode_calibration = profile_decode(self)
        from sglang.srt.multiplex.pp_memory import install as memory_install
        from sglang.srt.multiplex.pp_memory import record as memory_record

        memory_record("calibration", pp=self.pp_rank, tp=self.tp_rank)
        self._pp_receive_pool = None
        if self.pdmux_config.pp_prefetch and self.pp_rank:
            self._pp_receive_stream = torch.cuda.Stream()
            receive_tokens = max(self.max_prefill_tokens, self.model_config.context_len)
            chunk_tokens = self.server_args.pdmux_pp_prefill_chunk_size
            if chunk_tokens:
                # Every larger admission is split by the leader; even the
                # unchunked fast path is bounded by this wire-token budget.
                receive_tokens = min(receive_tokens, chunk_tokens)
            self._pp_receive_pool = PrefillReceivePool(
                receive_tokens,
                self.model_config.hidden_size,
                self.model_config.dtype,
                self.device,
                slots=self.pdmux_config.pp_prefetch_slots,
            )
            logger.info(
                "MuxWise-PP receive slots=%s max_tokens=%s reserved_MiB=%.1f",
                len(self._pp_receive_pool.buffers),
                receive_tokens,
                sum(
                    t.numel() * t.element_size()
                    for slot in self._pp_receive_pool.buffers
                    for t in slot.values()
                )
                / 2**20,
            )
        self._pp_decode_transport = None
        # TP uses the token-to-control protocol even with a legacy config.
        # TP1 retains its configured transport without extra collectives.
        if self.pdmux_config.pp_async_decode or self.tp_size > 1:
            from sglang.srt.multiplex.pp_decode import AsyncDecodeTransport

            max_bs = self.server_args.max_micro_batch_size or max(
                1, self.max_running_requests // self.pp_size
            )
            self._pp_decode_transport = AsyncDecodeTransport(
                self.pp_group,
                max_bs,
                self.model_config.hidden_size,
                self.model_config.dtype,
                self.device,
                token_leader=self.tp_rank == 0,
            )
        memory_record("slots", pp=self.pp_rank, tp=self.tp_rank)
        memory_install(self.tp_worker.model_runner)
        logger.info(
            "MuxWise-PP stage=%s layers=[%s,%s) capacity=%s partitions=%s radix=%s graphs=%s",
            self.pp_rank,
            self.tp_worker.model_runner.model.start_layer,
            self.tp_worker.model_runner.model.end_layer,
            self.max_total_num_tokens,
            len(self.stream_groups),
            (
                "disabled"
                if self.tree_cache.disable
                else self.server_args.pdmux_pp_radix_eviction
            ),
            (
                len(self.tp_worker.model_runner.graph_runner.graphs)
                if self.tp_worker.model_runner.graph_runner
                else 0
            ),
        )

    def _pp_tp_broadcast(self, value):
        if self.tp_size == 1:
            return value
        obj = [value]
        dist.broadcast_object_list(
            obj, src=self.tp_group.ranks[0], group=self.tp_cpu_group
        )
        return obj[0]

    def _pp_tp_all(self, ready):
        if self.tp_size == 1:
            return ready
        flag = torch.tensor([int(ready)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.tp_cpu_group)
        return bool(flag.item())

    def _pp_select_streams(self, idx):
        self._pp_models.last_switch = self._pp_decode_steps
        self._pp_stream_idx = idx
        self._pp_timing.switch(idx)
        set_current_stream_idx(idx)
        self._pp_prefill_stream, self._pp_decode_stream = self.stream_groups[idx]
        self.tp_worker.model_runner.update_decode_attn_backend(idx)

    def _pp_adjust_streams(self, decode_bs):
        if not decode_bs:
            decode_bs = max(mb.batch_size() for mb in self.running_mbs)
        idx = None
        if self.tp_rank == 0:
            idx = self.mux_pp_policy.partition(
                self.pdmux_config,
                len(self.stream_groups),
                decode_bs,
                (
                    local_prefill_runnable(self._pp_lane_works())
                    if self.server_args.pdmux_pp_runnable_partition
                    else any(w.state != "done" for w in self._pp_lane_works())
                ),
            )
        idx = self._pp_tp_broadcast(idx)
        if idx == self._pp_stream_idx:
            return
        # All TP peers must have retired their local transfers before switching.
        ready = all(
            (w.launched is None or w.launched.is_set())
            and (w.event is None or w.event.query())
            for w in self._pp_lane_works()
        )
        if not self._pp_tp_all(ready):
            return
        with self._pp_timing.region("prefill", "switch"), self._pp_timing.region(
            "decode", "switch"
        ):
            self._pp_prefill_stream.synchronize()
            self._pp_decode_stream.synchronize()
        if self.tp_size > 1:
            dist.barrier(group=self.tp_cpu_group)
        self._pp_select_streams(idx)
        logger.debug(
            "MuxWise-PP partition=%s decode_bs=%s SMs=%s",
            idx,
            decode_bs,
            self.sm_counts[idx],
        )

    def _pp_decode_notice(self, packet):
        if not self.server_args.pdmux_pp_decode_aware_slices or not self.cur_batch:
            return None
        bs, ctx = self.cur_batch.batch_size(), self.cur_batch.seq_lens_sum
        seconds = self._pp_models.decode_seconds(bs, ctx, self._pp_stream_idx)
        if not math.isfinite(seconds):
            return None
        now = time.monotonic()
        return dict(
            step=packet.step,
            mb=packet.mb_id,
            bs=bs,
            ctx=ctx,
            issued_at=now,
            seconds=seconds,
            eta=now + seconds,
            host=self._pp_clock_host,
        )

    def _pp_make_plan(self):
        stats = self._pp_admission_stats
        if not self.waiting_queue:
            stats.record("empty")
            return None
        limit = self._pp_max_prefills
        if self.server_args.pdmux_pp_adaptive_inflight:
            head = self.waiting_queue[0]
            limit = work_bounded_inflight_limit(
                self.pp_size,
                limit,
                self._pp_admission_budget,
                sum(w.batch.extend_num_tokens for w in self._pp_prefills.values()),
                getattr(head, "_pp_effective_tokens", len(head.origin_input_ids)),
            )
        token_limit = getattr(self.server_args, "pdmux_pp_max_inflight_tokens", 0)
        unfinished = (
            sum(
                w.batch.extend_num_tokens - w.completed_tokens
                for w in self._pp_prefills.values()
            )
            if token_limit
            else 0
        )
        if token_limit and unfinished >= token_limit:
            stats.record("inflight_tokens", tokens=unfinished, limit=token_limit)
            return None
        if not token_limit and len(self._pp_prefills) >= limit:
            stats.record("inflight", limit=limit, hard_limit=self._pp_max_prefills)
            return None
        extra_slot = (
            not token_limit
            and self.server_args.pdmux_pp_adaptive_inflight
            and len(self._pp_prefills) >= (3 * self.pp_size + 1) // 2
        )
        budget = self._pp_admission_budget
        if budget <= 0:
            stats.record("budget", budget=budget)
            return None
        refill = min(getattr(self.server_args, "pdmux_pp_token_refill_size", 0), budget)
        if token_limit and unfinished and token_limit - unfinished < refill:
            # A few newly returned credits can otherwise admit a tiny parent
            # batch on every control tick. Accumulate enough for useful work;
            # the empty-pipeline oversized-head escape and KV/output reserves
            # remain unchanged below. This gate never waits for new arrivals.
            stats.record(
                "inflight_refill", tokens=unfinished, limit=token_limit, refill=refill
            )
            return None
        # Reserve the entire remaining output, across ALL microbatches, plus
        # pp_size-1 safety tokens. Unmerged and ready-to-join requests count too.
        active = [r for mb in self.running_mbs for r in mb.reqs if not r.finished()]
        pending = [r for w in self._pp_prefills.values() for r in w.batch.reqs]
        pending += self._pp_pending_requests()
        active += pending  # input KV already allocated; reserve future output too
        reserve = (
            sum(
                max(0, r.sampling_params.max_new_tokens - len(r.output_ids))
                + self.pp_size
                - 1
                for r in active
            )
            if self.server_args.pdmux_pp_reserve_decode
            else 0
        )
        target = self.mux_pp_policy.merge_target(self.running_mbs)
        max_bs = self.server_args.max_micro_batch_size or max(
            1, self.max_running_requests // self.pp_size
        )
        if self.server_args.pdmux_pp_balance_merge:
            free_mb_slots = max_bs * self.pp_size - len(active)
        else:
            free_mb_slots = (
                max_bs
                - sum(not r.finished() for r in self.running_mbs[target].reqs)
                - len(pending)
            )
        slots = min(self.req_to_token_pool.available_size(), free_mb_slots)
        selected, probes = [], []
        input_tokens = 0
        reason, details = "admitted", dict(reserve=reserve, budget=budget)
        for req in self.waiting_queue:
            if len(selected) >= slots:
                reason = "max_running_requests"
                break
            req.init_next_round_input(self.tree_cache)
            probes.append((req.rid, len(req.fill_ids), len(req.prefix_indices)))
            req._pp_effective_tokens = req.extend_input_len
            self.tree_cache.inc_lock_ref(req.last_node)
            needed = (
                req.extend_input_len
                + req.sampling_params.max_new_tokens
                + self.pp_size
                - 1
            )
            available = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
            reason = stats.capacity_reason(available, needed, reserve)
            # Arrival-time prefix estimates may become stale after eviction.
            # Recheck the real locked probe before using an extra pipeline slot.
            if (
                reason is None
                and extra_slot
                and not selected
                and req.extend_input_len > budget
            ):
                reason = "inflight"
            if (
                reason is None
                and token_limit
                and (unfinished or selected)
                and unfinished + input_tokens + req.extend_input_len > token_limit
            ):
                reason = "inflight_tokens"
            if (
                reason is None
                and selected
                and input_tokens + req.extend_input_len > budget
            ):
                reason = "budget"
            if reason:
                details.update(
                    available_after_lock=available,
                    needed=needed,
                    reserve=reserve,
                    prefix=len(req.prefix_indices),
                )
                self.tree_cache.dec_lock_ref(req.last_node)
                break
            reason = "admitted"
            selected.append(req)
            reserve += needed
            input_tokens += req.extend_input_len
        plan = PrefillPlan(self._pp_batch_id, probes, [r.rid for r in selected])
        if selected:
            chunk_size = getattr(self.server_args, "pdmux_pp_prefill_chunk_size", 0)
            if chunk_size and input_tokens > chunk_size:
                plan.chunks = plan_chunks(
                    [r.extend_input_len for r in selected], chunk_size
                )
            self._pp_batch_id += 1
            self._pp_build_prefill(plan, selected)
        stats.record(reason, len(selected), **details)
        return plan  # Failed probes also have to be replayed.

    def _pp_apply_plan(self, plan):
        req_by_id = {r.rid: r for r in self.waiting_queue}
        selected_ids = set(plan.rids)
        selected = []
        for rid, fill_len, prefix_len in plan.probes:
            req = req_by_id[rid]
            req.init_next_round_input(self.tree_cache)
            assert (len(req.fill_ids), len(req.prefix_indices)) == (
                fill_len,
                prefix_len,
            ), (
                f"PP radix divergence stage={self.pp_rank} rid={rid} "
                f"expected={(fill_len, prefix_len)} actual={(len(req.fill_ids), len(req.prefix_indices))}"
            )
            if rid in selected_ids:
                self.tree_cache.inc_lock_ref(req.last_node)
                selected.append(req)
        assert [r.rid for r in selected] == plan.rids
        if selected:
            self._pp_build_prefill(plan, selected)

    def _pp_build_prefill(self, plan, reqs):
        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            False,
            self.spec_algorithm,
        )
        batch.prepare_for_extend()
        # The CPU already knows these lengths. Re-reading seq_lens from CUDA
        # on the first split would synchronize prefill with the decode lane,
        # including small whole batches that do not need CPP children.
        if self.server_args.pdmux_pp_cpu_prefill_metadata:
            batch.pdmux_pp_cpu_metadata = True
            batch.pdmux_pp_seq_lens_cpu = torch.tensor(
                [p + n for p, n in zip(batch.prefix_lens, batch.extend_lens)],
                dtype=torch.int64,
            )
        batch.forward_mode = ForwardMode.SPLIT_PREFILL
        batch.split_index = self.tp_worker.model_runner.model.start_layer
        self.waiting_queue = [
            r for r in self.waiting_queue if r.rid not in set(plan.rids)
        ]
        event = self._pp_decode_stream.record_event()
        self._pp_prefill_stream.wait_event(event)
        self._pp_prefills[plan.batch_id] = PrefillWork(
            plan, batch, "compute" if self.pp_rank == 0 else "waiting", event
        )
        parent = self._pp_prefills[plan.batch_id]
        for i, chunk in enumerate(plan.chunks):
            child = chunk_batch(batch, chunk)
            # READY uses an int64 wire ID. Negative IDs keep chunks disjoint
            # from small whole batches, which retain the original fast path.
            wire_id = -((plan.batch_id << 32) + i + 1)
            child_plan = PrefillPlan(wire_id, [], [r.rid for r in child.reqs])
            parent.chunks.append(
                PrefillWork(
                    child_plan,
                    child,
                    parent.state,
                    event,
                    parent_id=plan.batch_id,
                    chunk_index=i,
                )
            )
        if parent.chunks:
            # Child sequence lengths and sampling tensors were allocated on
            # the control/decode stream after the parent's allocation fence.
            # Prefill must also wait for those copies before planning attention.
            event = self._pp_decode_stream.record_event()
            self._pp_prefill_stream.wait_event(event)
            parent.event = event
            for child in parent.chunks:
                child.event = event
        self._pp_timing.request_event(
            "admit",
            self._pp_prefills[plan.batch_id],
            tokens=batch.extend_num_tokens,
            cached=sum(len(r.prefix_indices) for r in reqs),
        )
        self._pp_cached_tokens += sum(len(r.prefix_indices) for r in reqs)
        logger.debug(
            "MuxWise-PP admit batch=%s requests=%s tokens=%s cached=%s inflight=%s",
            plan.batch_id,
            len(reqs),
            batch.extend_num_tokens,
            sum(len(r.prefix_indices) for r in reqs),
            len(self._pp_prefills),
        )

    def _pp_progress_prefill(self):
        try:
            self._pp_progress_prefill_impl()
        finally:
            self._pp_timing.prefill_state()

    def _pp_begin_receive(self, work, envelope):
        assert envelope["plan"] == work.plan
        pool = getattr(self, "_pp_receive_pool", None)
        if pool is None:
            tensors = {
                k: torch.empty(shape, dtype=dtype, device=self.device)
                for k, shape, dtype in envelope["tensors"]
            }
            stream = self._pp_prefill_stream
        else:
            acquired = pool.acquire(envelope["tensors"])
            if acquired is None:
                if not getattr(work, "trace_slot_wait", False):
                    self._pp_timing.request_event("receive_slot_wait", work)
                    work.trace_slot_wait = True
                return False
            work.recv_slot, tensors = acquired
            stream = self._pp_receive_stream
        work.proxy = PPProxyTensors(tensors)
        work.launched, work.event = self.prefill_mailbox.transfer(
            False, tensors, stream
        )
        work.state = "receiving"
        self._pp_timing.request_event("receive_posted", work)
        self.prefill_mailbox.outbox.put(dict(kind="ready", batch_id=work.plan.batch_id))
        return True

    def _pp_post_receives(self):
        for work in self._pp_lane_works():
            if work.state != "waiting":
                continue
            envelope = self._pp_envelopes.get(work.plan.batch_id)
            if not self._pp_tp_all(envelope is not None):
                break
            if not self._pp_begin_receive(work, envelope):
                break
            del self._pp_envelopes[work.plan.batch_id]

    def _pp_release_receive(self, work):
        if work.recv_slot is not None:
            self._pp_receive_pool.release(work.recv_slot)
            work.recv_slot = None

    def _pp_progress_prefill_impl(self):
        while (envelope := self.prefill_mailbox.poll()) is not None:
            if envelope["kind"] == "status":
                if self.pp_rank == 0:
                    self._pp_stage_status[envelope["stage"]] = envelope["data"]
                else:
                    self.prefill_mailbox.outbox.put(envelope)
                continue
            bid = envelope["batch_id"]
            if envelope["kind"] == "chunk_done":
                assert self.pp_rank == 0
                parent = self._pp_prefills[bid]
                assert envelope["chunk_index"] == parent.acked_chunks
                assert (
                    envelope["tokens"] == parent.plan.chunks[parent.acked_chunks].tokens
                )
                parent.acked_chunks += 1
                parent.completed_tokens += envelope["tokens"]
                assert parent.completed_tokens <= parent.batch.extend_num_tokens
                continue
            if envelope["kind"] == "ready":
                self.prefill_mailbox.ready.add(bid)
                continue
            if self.pp_rank == 0:
                assert envelope["kind"] == "result" and bid in self._pp_prefills
                parent = self._pp_prefills[bid]
                if parent.chunks:
                    assert parent.acked_chunks == len(parent.chunks)
                    assert parent.completed_tokens == parent.batch.extend_num_tokens
                else:
                    parent.completed_tokens = parent.batch.extend_num_tokens
                self._pp_timing.request_event("result_received", parent)
                self._pp_merge_ready.append(
                    dict(
                        batch_id=bid,
                        target_mb=self.mux_pp_policy.merge_target(self.running_mbs),
                        next_token_ids=envelope["tokens"],
                    )
                )
            else:
                assert bid not in self._pp_envelopes
                self._pp_timing.request_event(
                    "hidden_metadata", rids=envelope["plan"].rids, batch=bid
                )
                self._pp_envelopes[bid] = envelope
        # Keep compute FIFO while allowing at most one additional input to arrive.
        # A sent batch retains its KV ownership until the ordered MERGE.
        if getattr(self, "_pp_receive_pool", None) is not None:
            self._pp_post_receives()
        work = next((w for w in self._pp_lane_works() if w.state != "done"), None)
        self.split_prefill_batch = work.batch if work else None
        if work is None:
            return
        if not getattr(work, "trace_head", False):
            self._pp_timing.request_event("fifo_head", work)
            work.trace_head = True
        if work.state == "waiting":
            envelope = self._pp_envelopes.get(work.plan.batch_id)
            if not self._pp_tp_all(envelope is not None):
                return
            if not self._pp_begin_receive(work, envelope):
                return
            del self._pp_envelopes[work.plan.batch_id]
        if work.state == "offering" and work.launched is None:
            if not self._pp_tp_all(work.plan.batch_id in self.prefill_mailbox.ready):
                return
            self.prefill_mailbox.ready.remove(work.plan.batch_id)
            work.launched, work.event = self.prefill_mailbox.transfer(
                True, work.proxy.tensors, self._pp_prefill_stream
            )
            work.state = "sending"
        if work.launched is not None:
            if not self._pp_tp_all(work.launched.is_set()):
                return
            work.launched = None
            if work.state == "offering":
                self._pp_timing.request_event("send_launched", work)
                work.state = "sending"
        if not self._pp_tp_all(work.event is None or work.event.query()):
            return
        if work.trace_slice:
            self._pp_timing.request_event(
                "slice_observed_done", work, layer=work.batch.split_index
            )
            work.trace_slice = False
            if work.batch.split_index == self.tp_worker.model_runner.model.end_layer:
                self._pp_timing.request_event("prefill_done", work)
        if work.state == "sending":
            self._pp_timing.request_event("send_done", work)
            work.state, work.proxy = "done", None
            work.batch.split_forward_batch = None
            self._pp_release_receive(work)
            self.split_prefill_batch = None
            self._pp_finish_chunk(work)
            return
        if work.state == "receiving":
            self._pp_timing.request_event("hidden_received", work)
            self._pp_prefill_stream.wait_event(work.event)
            work.state = "compute"
        batch = work.batch
        end = self.tp_worker.model_runner.model.end_layer
        if batch.split_index < end:
            # ADMIT may have arrived after this step's partition decision.
            # Keep decode progressing until the next safe mixed/P-only switch.
            if self.sm_counts[self._pp_stream_idx][0] == 0:
                return
            count = self.mux_pp_policy.slice_layers(
                self.pdmux_config, batch.extend_num_tokens, end - batch.split_index
            )
            base_count = count
            if self.server_args.pdmux_pp_decode_aware_slices and self.pp_rank:
                count = self._pp_tp_broadcast(
                    self._pp_models.arrival_slice_layers(
                        count,
                        self._pp_upstream_decode,
                        time.monotonic(),
                        self._pp_clock_host,
                    )
                    if self.tp_rank == 0
                    else None
                )
                assert 1 <= count <= base_count
                self._pp_slice_choices[0] += 1
                self._pp_slice_choices[1] += count < base_count
                self._pp_slice_choices[2] += base_count - count
            if not work.trace_started:
                self._pp_timing.request_event("prefill_start", work)
                work.trace_started = True
            self._pp_timing.request_event(
                "slice_launch",
                work,
                layer=batch.split_index,
                base_layers=base_count,
                upstream_decode=self._pp_upstream_decode,
                layers=count,
                partition=self._pp_stream_idx,
            )
            work.trace_slice = True
            batch.split_forward_count = count
            self._pp_timing.mark("prefill", "compute_slot")
            result = self.run_batch(batch, pp_proxy_tensors=work.proxy)
            batch.split_index += count
            self._pp_prefill_steps += 1
            if batch.split_index == end:
                batch.split_prefill_finished = True
                work.proxy = (
                    PPProxyTensors(result.pp_hidden_states_proxy_tensors)
                    if not self.pp_group.is_last_rank
                    else None
                )
                work.tokens = result.next_token_ids
            work.event = self._pp_prefill_stream.record_event()
            return
        if self.pp_group.is_last_rank:
            if work.parent_id is not None:
                parent = self._pp_prefills[work.parent_id]
                chunk = parent.plan.chunks[work.chunk_index]
                finals = work.batch.pdmux_pp_final_indices
                if finals:
                    tokens = work.tokens.tolist()
                    assert len(tokens) == len(finals)
                    for j, token in zip(finals, tokens):
                        parent.sampled_tokens[chunk.spans[j][0]] = token
                if self.tp_rank == 0:
                    self.prefill_mailbox.outbox.put(
                        dict(
                            kind="chunk_done",
                            batch_id=work.parent_id,
                            chunk_index=work.chunk_index,
                            tokens=chunk.tokens,
                        )
                    )
                final = work.chunk_index == len(parent.chunks) - 1
                if final:
                    assert len(parent.sampled_tokens) == len(parent.batch.reqs)
                    tokens = [
                        parent.sampled_tokens[i] for i in range(len(parent.batch.reqs))
                    ]
                bid = work.parent_id
            else:
                final, bid = True, work.plan.batch_id
                tokens = work.tokens.tolist() if self.tp_rank == 0 else None
            if self.tp_rank == 0 and final:
                self._pp_timing.request_event("result_send", work)
                self.prefill_mailbox.outbox.put(
                    dict(
                        kind="result",
                        batch_id=bid,
                        tokens=tokens,
                    )
                )
            work.state = "done"
            work.batch.split_forward_batch = None
            self._pp_release_receive(work)
            self.split_prefill_batch = None
            self._pp_finish_chunk(work)
        else:
            tensors = work.proxy.tensors
            self._pp_timing.request_event("hidden_offer", work)
            if self.pdmux_config.pp_prefetch:
                work.launched, work.event = self.prefill_mailbox.send_when_ready(
                    work.plan.batch_id, tensors, self._pp_prefill_stream
                )
            self.prefill_mailbox.outbox.put(
                dict(
                    kind="hidden",
                    batch_id=work.plan.batch_id,
                    plan=work.plan,
                    tensors=[(k, tuple(t.shape), t.dtype) for k, t in tensors.items()],
                )
            )
            # Do not post an unmatched send: cudaMalloc/cudaFree in a later
            # control step can synchronize that stream before the receiver has
            # allocated its buffer. Receiver READY makes this dependency acyclic.
            work.state = "offering"

    def _pp_finish_chunk(self, work):
        if work.parent_id is not None:
            parent = self._pp_prefills[work.parent_id]
            if work.chunk_index == len(parent.chunks) - 1:
                parent.state = "done"
                parent.batch.split_prefill_finished = True
                parent.event = work.event

    def _pp_merge(self, merge, mb_id):
        work = self._pp_prefills[merge["batch_id"]]
        self._pp_timing.request_event("merge", work, mb=mb_id)
        assert merge["target_mb"] == mb_id
        batch = work.batch
        if work.chunks:
            # A result can outrun polling of the last local send. Its launch
            # and receive-slot fences must retire before deleting the parent.
            for child in work.chunks:
                if child.launched is not None:
                    while not child.launched.wait(timeout=0.1):
                        if not self.prefill_mailbox.errors.empty():
                            raise RuntimeError(
                                "Chunk send failed before MERGE"
                            ) from self.prefill_mailbox.errors.get()
                self._pp_decode_stream.wait_event(child.event)
                if child.recv_slot is not None:
                    self._pp_receive_stream.wait_event(child.event)
                    self._pp_release_receive(child)
                child.batch.split_forward_batch = None
            assert all(child.batch.split_prefill_finished for child in work.chunks)
            batch.split_prefill_finished = True
            work.event = work.chunks[-1].event
        assert batch.split_prefill_finished
        # The last-stage result can outrun local polling of a completed send.
        # Do not delete its receive-slot owner before recording the reuse fence.
        if work.launched is not None:
            while not work.launched.wait(timeout=0.1):
                if not self.prefill_mailbox.errors.empty():
                    raise RuntimeError(
                        "Prefill send failed before MERGE"
                    ) from self.prefill_mailbox.errors.get()
        self._pp_decode_stream.wait_event(work.event)
        if work.recv_slot is not None:
            self._pp_receive_stream.wait_event(work.event)
            self._pp_release_receive(work)
        if self.server_args.pdmux_pp_cpu_merge_tokens:
            # The first tokens are already CPU control data. Like decode
            # retirement, retain the GPU copy for future forward/split_batch,
            # but process/cache/emit the CPU values at the same MERGE point.
            cpu_ids = torch.tensor(merge["next_token_ids"], dtype=torch.int64)
            batch.output_ids = cpu_ids.to(self.device, non_blocking=True)
            self.process_batch_result(batch, self._pp_result(cpu_ids))
        else:
            batch.output_ids = torch.tensor(
                merge["next_token_ids"], device=self.device, dtype=torch.int64
            )
            self.process_batch_result(batch, self._pp_result(batch.output_ids))
        batch.split_forward_batch = (
            None  # Release hidden/residual and attention metadata.
        )
        batch.filter_batch()
        targets = dict(zip(work.plan.rids, merge["targets"]))
        remaining = batch if not batch.is_empty() else None
        for target in range(self.pp_size):
            if remaining is None:
                break
            indices = [
                i for i, r in enumerate(remaining.reqs) if targets[r.rid] == target
            ]
            if not indices:
                continue
            part, remaining = remaining.split_batch(indices)
            self._pp_pending_merges[target].append(part)
        del self._pp_prefills[merge["batch_id"]]
        if self.split_prefill_batch is batch:
            self.split_prefill_batch = None
        if self._pp_leader:
            assert self._pp_merge_ready.popleft()["batch_id"] == merge["batch_id"]

    def _pp_pending_requests(self):
        return [
            r
            for parts in self._pp_pending_merges
            for batch in parts
            for r in batch.reqs
        ]

    def _pp_join_pending(self, mb_id):
        batch = self.running_mbs[mb_id]
        batch.filter_batch()
        for part in self._pp_pending_merges[mb_id]:
            if batch.is_empty():
                batch = part
            else:
                batch.merge_batch(part)
        self._pp_pending_merges[mb_id] = []
        self.running_mbs[mb_id] = self.running_batch = batch

    def _pp_merge_targets(self, merge):
        reqs = self._pp_prefills[merge["batch_id"]].batch.reqs
        args = self.server_args
        max_bs = args.max_micro_batch_size or max(
            1, self.max_running_requests // self.pp_size
        )
        if not args.pdmux_pp_balance_merge:
            candidates = [
                i
                for i, mb in enumerate(self.running_mbs)
                if sum(not r.finished() for r in mb.reqs)
                + len(reqs)
                + sum(len(b.reqs) for b in self._pp_pending_merges[i])
                <= max_bs
            ]
            if not candidates:
                return False
            merge["target_mb"] = min(
                candidates, key=lambda i: self.running_mbs[i].batch_size()
            )
            merge["targets"] = [merge["target_mb"]] * len(reqs)
        else:
            merge["targets"] = balanced_assignment(
                reqs,
                [mb.reqs for mb in self.running_mbs],
                [
                    [r for b in parts for r in b.reqs]
                    for parts in self._pp_pending_merges
                ],
                args.pdmux_pp_load_bs_weight,
                args.pdmux_pp_load_ctx_weight,
                max_bs,
            )
        return True

    def _pp_rebalance_control(self, packet, in_flight):
        args = self.server_args
        if not args.pdmux_pp_rebalance:
            return
        a, b, unit = (
            args.pdmux_pp_load_bs_weight,
            args.pdmux_pp_load_ctx_weight,
            args.pdmux_pp_rebalance_unit,
        )
        if not self._pp_hold and packet.step % args.pdmux_pp_rebalance_interval == 0:
            pair = rebalance_pair(
                [
                    mb.reqs
                    + [r for batch in self._pp_pending_merges[i] for r in batch.reqs]
                    for i, mb in enumerate(self.running_mbs)
                ],
                a,
                b,
                unit,
            )
            if pair:
                self._pp_hold = list(pair)
        if self._pp_hold and all(in_flight[i] is None for i in self._pp_hold):
            source, target = self._pp_hold
            max_bs = args.max_micro_batch_size or max(
                1, self.max_running_requests // self.pp_size
            )
            rids = migration_rids(
                self.running_mbs[source].reqs,
                self.running_mbs[target].reqs
                + [r for batch in self._pp_pending_merges[target] for r in batch.reqs],
                a,
                b,
                unit,
                max_bs,
            )
            if rids:
                packet.migrations = [MigrationPlan(rids, source, target)]
            self._pp_hold = []
        packet.hold_mbs = self._pp_hold.copy()

    def _pp_apply_migrations(self, packet, in_flight):
        self._pp_hold = packet.hold_mbs.copy()
        for plan in packet.migrations:
            source, target = plan.source_mb, plan.target_mb
            assert in_flight[source] is None and in_flight[target] is None
            src, dst = self.running_mbs[source], self.running_mbs[target]
            src.filter_batch()
            dst.filter_batch()
            ids = set(plan.rids)
            move_indices = [i for i, r in enumerate(src.reqs) if r.rid in ids]
            assert len(move_indices) == len(ids)
            moved, kept = src.split_batch(move_indices)
            self.running_mbs[source] = kept or ScheduleBatch(reqs=[])
            if dst.is_empty():
                self.running_mbs[target] = moved
            else:
                dst.merge_batch(moved)
            self._pp_migrations += len(ids)
            logger.debug(
                "MuxWise-PP migrate step=%s source=%s target=%s rids=%s",
                packet.step,
                source,
                target,
                plan.rids,
            )
        self.running_batch = self.running_mbs[packet.mb_id]

    def _pp_effective_input(self, req):
        # Read-only prefix length: arrivals must not split/access radix nodes
        # outside the replicated admission probes.
        if self.tree_cache.disable:
            return len(req.origin_input_ids)
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(req.origin_input_ids[:-1], req.extra_key)
        node, matched = self.tree_cache.root_node, 0
        while len(key):
            node = node.children.get(self.tree_cache.get_child_key_fn(key))
            if node is None:
                break
            n = self.tree_cache.key_match_fn(node.key, key)
            matched += n
            if n < len(node.key):
                break
            key = key[n:]
        return len(req.origin_input_ids) - matched

    def _pp_update_demand(self, packet):
        args = self.server_args
        now = time.monotonic()
        if self._pp_leader:
            for r in self.waiting_queue:
                if not hasattr(r, "_pp_arrival"):
                    r._pp_arrival = now
                    r._pp_effective_tokens = self._pp_effective_input(r)
                    self._pp_arrivals.append((now, r._pp_effective_tokens))
            while (
                self._pp_arrivals
                and self._pp_arrivals[0][0] < now - args.pdmux_pp_arrival_window
            ):
                self._pp_arrivals.popleft()
            waiting = [
                (r._pp_arrival, r._pp_effective_tokens) for r in self.waiting_queue
            ]
            self._pp_required = required_throughput(
                waiting,
                self._pp_arrivals,
                now,
                args.pdmux_pp_slo_ttft,
                args.pdmux_pp_slo_tpot,
                args.pdmux_pp_fresh_eps,
                args.pdmux_pp_arrival_window,
            )
            self._pp_waiting_tokens = sum(n for _, n in waiting)
            if not args.pdmux_pp_lazy_admission_budget or (
                self.waiting_queue and len(self._pp_prefills) < self._pp_max_prefills
            ):
                started = time.monotonic()
                self._pp_admission_budget = self.mux_pp_policy.admission_budget(self)
                self._pp_budget_stats["calls"] += 1
                self._pp_budget_stats["seconds"] += time.monotonic() - started
            else:
                self._pp_budget_stats["skipped"] += 1
            packet.required, packet.waiting_tokens, packet.admission_budget = (
                self._pp_required,
                self._pp_waiting_tokens,
                self._pp_admission_budget,
            )
        else:
            self._pp_required, self._pp_waiting_tokens, self._pp_admission_budget = (
                packet.required,
                packet.waiting_tokens,
                packet.admission_budget,
            )

    def _pp_summary(self):
        if self.tp_rank != 0:
            return
        now = time.monotonic()
        if now - self._pp_last_summary < self.server_args.pdmux_pp_summary_interval:
            return
        data = dict(
            partition=self._pp_stream_idx,
            slice_choices=self._pp_slice_choices.copy(),
            admission_budget_eval=self._pp_budget_stats.copy(),
            samples=[self._pp_models.decode.count, self._pp_models.prefill.count],
            mape=[self._pp_models.decode.mape, self._pp_models.prefill.mape],
            prefill_slice_mape=self._pp_models.prefill_slice.mape,
            prefill_legacy_batch_mape=(
                sum(self._pp_models.prefill_legacy_batch_errors)
                / len(self._pp_models.prefill_legacy_batch_errors)
                if self._pp_models.prefill_legacy_batch_errors
                else None
            ),
            by_partition=[row.copy() for row in self._pp_models.samples_by_partition],
            cuda_memory=dict(
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                max_allocated=torch.cuda.max_memory_allocated(),
                max_reserved=torch.cuda.max_memory_reserved(),
            ),
            timing=self._pp_timing.snapshot(),
            admission=(
                self._pp_admission_stats.snapshot() if self.pp_rank == 0 else None
            ),
            at=now,
        )
        if self.pp_rank:
            self._pp_last_summary = now
            self.prefill_mailbox.outbox.put(
                dict(kind="status", stage=self.pp_rank, data=data)
            )
            return
        if (
            0 in self._pp_stage_status
            and now - self._pp_stage_status[0]["at"]
            < self.server_args.pdmux_pp_summary_interval
        ):
            data["admission"]["series"] = (
                self._pp_stage_status[0]["admission"]["series"]
                + data["admission"]["series"]
            )
        self._pp_stage_status[0] = data
        # Give downstream reports a control turn to arrive before logging;
        # include all stages from this reporting interval, not the previous one.
        if len(self._pp_stage_status) != self.pp_size or any(
            now - v["at"] > self.server_args.pdmux_pp_summary_interval / 2
            for v in self._pp_stage_status.values()
        ):
            return
        self._pp_last_summary = now
        logger.info(
            "MuxWise-PP summary stages=%s mbs=%s waiting=%s/%s required=%s budget=%s inflight=%s pending=%s migrations=%s",
            self._pp_stage_status,
            [
                (len(mb.reqs), sum(req_context(r) for r in mb.reqs))
                for mb in self.running_mbs
            ],
            len(self.waiting_queue),
            self._pp_waiting_tokens,
            self._pp_required,
            self._pp_admission_budget,
            len(self._pp_prefills),
            len(self._pp_pending_requests()),
            self._pp_migrations,
        )

    def _pp_abort_ids(self, requests):
        aborts = [r for r in requests if isinstance(r, AbortReq)]
        reqs = self.waiting_queue + [r for mb in self.running_mbs for r in mb.reqs]
        reqs += [r for w in self._pp_prefills.values() for r in w.batch.reqs]
        reqs += self._pp_pending_requests()
        return sorted(
            {
                r.rid
                for r in reqs
                for a in aborts
                if a.abort_all or r.rid.startswith(a.rid)
            }
        )

    def _pp_apply_aborts(self, rids):
        from sglang.srt.disaggregation.utils import prepare_abort

        ids = set(rids)
        for req in self.waiting_queue:
            if req.rid in ids:
                # A terminal batch output also removes tokenizer-side request
                # state; the upstream bare AbortReq path only wakes the waiter.
                prepare_abort(req, "Aborted before prefill")
                self.stream_output([req], req.return_logprob)
        self.waiting_queue = [r for r in self.waiting_queue if r.rid not in ids]
        reqs = [r for mb in self.running_mbs for r in mb.reqs]
        reqs += [r for w in self._pp_prefills.values() for r in w.batch.reqs]
        reqs += self._pp_pending_requests()
        for req in reqs:
            if req.rid in ids and not req.finished():
                req.to_abort = True

    def _pp_retract(self, rids):
        batch = self.running_batch
        removed = []
        for rid in rids:
            idx = next(i for i, r in enumerate(batch.reqs) if r.rid == rid)
            removed.append(batch.reqs[idx])
            batch.release_req(idx, len(batch.reqs) - len(removed), self.server_args)
        if removed:
            batch.filter_batch(
                keep_indices=[i for i, r in enumerate(batch.reqs) if r not in removed]
            )
            self._extend_requests_to_queue(removed, is_retracted=True)
            self.total_retracted_reqs += len(removed)

    @torch.inference_mode()
    def event_loop_pdmux_pp(self):
        self._pp_init()
        in_flight = [None] * self.pp_size
        pp_outputs = None
        transport = self._pp_decode_transport
        flight_steps = [None] * self.pp_size
        step = 0
        try:
            while True:
                for mb_id in range(self.pp_size):
                    self.running_batch = self.running_mbs[mb_id]
                    self.cur_batch = None
                    self.last_batch = None
                    self._pp_timing.mark("decode", "control")
                    self._pp_timing.drain()
                    self._pp_adjust_streams(self.running_batch.batch_size())
                    with torch.cuda.stream(self._pp_decode_stream):
                        set_pdmux_status(False)
                        if self.pp_rank == 0:
                            requests = self.recv_requests()
                            self._pp_timing.request_event(
                                "arrival",
                                rids=[r.rid for r in requests if hasattr(r, "rid")],
                            )
                            packet = PPControl(step, mb_id, requests=requests)
                            if (
                                transport
                                and self.tp_rank == 0
                                and in_flight[mb_id] is not None
                            ):
                                self._pp_timing.mark("decode", "token")
                                packet.retired = transport.receive_tokens(
                                    flight_steps[mb_id], mb_id
                                )
                            # The leader publishes only after all allocator
                            # operations have completed its eviction journal.
                            if self.tp_rank != 0:
                                packet = self._pp_tp_broadcast(None)
                        elif transport:
                            self._pp_timing.mark("decode", "control_recv")
                            packet, proxy_receive = transport.receive_control()
                            assert (packet.step, packet.mb_id) == (step, mb_id)
                        else:
                            obj = [None]
                            self._pp_timing.mark("decode", "control_recv")
                            dist.recv_object_list(
                                obj,
                                src=self.pp_group.ranks[self.pp_rank - 1],
                                group=self.world_group.cpu_group,
                            )
                            self._pp_timing.mark("decode", "control")
                            packet = obj[0]
                            assert (packet.step, packet.mb_id) == (step, mb_id)
                        assert (packet.step, packet.mb_id) == (step, mb_id)
                        self._pp_upstream_decode = (
                            packet.decode_notice if self.pp_rank else None
                        )
                        if transport:
                            transport.check()
                            if packet.retired is not None:
                                self._pp_timing.mark("decode", "retire")
                                retired = packet.retired
                                assert (retired["step"], retired["mb_id"]) == (
                                    flight_steps[mb_id],
                                    mb_id,
                                )
                                batch = in_flight[mb_id]
                                assert (
                                    batch is not None
                                    and len(retired["tokens"]) == batch.batch_size()
                                )
                                cpu_ids = torch.tensor(
                                    retired["tokens"], dtype=torch.int64
                                )
                                batch.output_ids = cpu_ids.to(
                                    self.device, non_blocking=True
                                )
                                self.process_batch_result(
                                    batch,
                                    self._pp_result(
                                        cpu_ids, retired["can_run_cuda_graph"]
                                    ),
                                )
                                in_flight[mb_id] = None
                            self._pp_timing.mark("decode", "control")
                        self._pp_eviction.begin(packet.evictions)
                        # Abort decisions include all microbatches and in-flight prefill.
                        self.process_input_requests(
                            [r for r in packet.requests if not isinstance(r, AbortReq)]
                        )
                        if self._pp_leader:
                            packet.abort = self._pp_abort_ids(packet.requests)
                        self._pp_apply_aborts(packet.abort)
                        assert in_flight[mb_id] is None
                        if self._pp_leader and self._pp_merge_ready:
                            merge = self._pp_merge_ready[0]
                            if (
                                self._pp_merge_targets(merge)
                                and merge["target_mb"] == mb_id
                            ):
                                packet.merge = merge
                        if packet.merge:
                            self._pp_merge(packet.merge, mb_id)
                        self._pp_join_pending(mb_id)
                        if self._pp_leader:
                            self._pp_rebalance_control(packet, in_flight)
                        self._pp_apply_migrations(packet, in_flight)
                        self._pp_update_demand(packet)
                        if self._pp_leader:
                            packet.plan = self._pp_make_plan()
                        elif packet.plan:
                            self._pp_apply_plan(packet.plan)
                        self.running_batch.filter_batch()
                        # Only stage zero decides retraction. Full output reservation
                        # normally makes this branch unnecessary; keep an explicit command.
                        if self._pp_leader and not self.running_batch.is_empty():
                            available = (
                                self.token_to_kv_pool_allocator.available_size()
                                + self.tree_cache.evictable_size()
                            )
                            if (
                                available
                                < self.running_batch.batch_size() * self.page_size
                            ):
                                packet.retract = [
                                    r.rid for r in self.running_batch.reqs
                                ]
                        self._pp_retract(packet.retract)
                        if (
                            not self.running_batch.is_empty()
                            and mb_id not in self._pp_hold
                        ):
                            self.running_batch.prepare_for_decode()
                            self.cur_batch = self.running_batch
                        self.running_mbs[mb_id] = self.running_batch
                        in_flight[mb_id] = self.cur_batch
                        self._pp_eviction.end()
                        if self._pp_leader:
                            packet.decode_bs = (
                                self.cur_batch.batch_size() if self.cur_batch else 0
                            )
                        if self._pp_leader:
                            packet.decode_notice = self._pp_decode_notice(packet)
                        if self._pp_leader and self.tp_size > 1:
                            self._pp_tp_broadcast(packet)
                        if transport:
                            flight_steps[mb_id] = step if self.cur_batch else None
                            bs = self.cur_batch.batch_size() if self.cur_batch else 0
                            if self._pp_leader:
                                packet.decode_bs = bs
                            else:
                                assert (
                                    packet.decode_bs == bs
                                ), "PP decode batch divergence"
                            if not self.pp_group.is_last_rank:
                                # Includes proxy shape and retired tokens; prefetch on
                                # the next stage while this stage computes.
                                if (
                                    self.pp_rank
                                    and self.server_args.pdmux_pp_decode_aware_slices
                                ):
                                    packet.decode_notice = self._pp_tp_broadcast(
                                        self._pp_decode_notice(packet)
                                        if self.tp_rank == 0
                                        else None
                                    )
                                transport.send_control(packet)
                    with torch.cuda.stream(self._pp_prefill_stream):
                        set_pdmux_status(True)
                        self._pp_timing.mark("decode", "prefill_launch")
                        previous_slices = self._pp_prefill_steps
                        self._pp_progress_prefill()
                        if self.cur_batch and self._pp_prefill_steps > previous_slices:
                            self._pp_overlap_launches += 1
                    with torch.cuda.stream(self._pp_decode_stream):
                        set_pdmux_status(False)
                        result = None
                        self._pp_timing.mark(
                            "decode", "empty" if not self.cur_batch else "launch"
                        )
                        if self.cur_batch:
                            self._pp_timing.decode_launch(mb_id, self.cur_batch)
                            proxy = None
                            if transport and self.pp_rank:
                                self._pp_timing.mark("decode", "proxy")
                                proxy_receive.wait()
                                proxy = PPProxyTensors(proxy_receive.payload)
                            result = self.run_batch(
                                self.cur_batch, pp_proxy_tensors=proxy
                            )
                            self._pp_decode_steps += 1
                        if transport:
                            self._pp_timing.mark("decode", "send")
                            if self.cur_batch:
                                if self.pp_group.is_last_rank and self.tp_rank == 0:
                                    transport.send_tokens(
                                        step,
                                        mb_id,
                                        result.next_token_ids,
                                        result.can_run_cuda_graph,
                                    )
                                elif not self.pp_group.is_last_rank:
                                    transport.send_proxy(
                                        mb_id,
                                        result.pp_hidden_states_proxy_tensors,
                                        self.cur_batch.batch_size(),
                                    )
                                if self.pp_rank:
                                    # Residual may alias the receive slot; include
                                    # the owned send copy in its consumption fence.
                                    transport.release_proxy(mb_id)
                        else:
                            # This is the upstream PP ring: retire the next microbatch,
                            # relay the previous token result, then send controls/proxy.
                            self._pp_timing.mark("decode", "send")
                            if self.pp_group.is_last_rank and self.cur_batch:
                                pp_outputs = {
                                    "next_token_ids": result.next_token_ids,
                                    "can_run_cuda_graph": result.can_run_cuda_graph,
                                }
                                self.pp_group.send_tensor_dict(
                                    pp_outputs, all_gather_group=self.attn_tp_group
                                )
                            next_mb = (mb_id + 1) % self.pp_size
                            next_outputs = None
                            if in_flight[next_mb] is not None:
                                self._pp_timing.mark("decode", "token")
                                next_outputs = self.pp_group.recv_tensor_dict(
                                    all_gather_group=self.attn_tp_group
                                )
                                self._pp_timing.mark("decode", "retire")
                                batch = in_flight[next_mb]
                                batch.output_ids = next_outputs["next_token_ids"]
                                self.process_batch_result(
                                    batch,
                                    self._pp_result(
                                        batch.output_ids,
                                        next_outputs["can_run_cuda_graph"],
                                    ),
                                )
                                in_flight[next_mb] = None
                            self._pp_timing.mark("decode", "send")
                            if not self.pp_group.is_last_rank:
                                if pp_outputs is not None:
                                    self.pp_group.send_tensor_dict(
                                        pp_outputs, all_gather_group=self.attn_tp_group
                                    )
                                dist.send_object_list(
                                    [packet],
                                    dst=self.pp_group.ranks[self.pp_rank + 1],
                                    group=self.world_group.cpu_group,
                                )
                                if self.cur_batch:
                                    self.pp_group.send_tensor_dict(
                                        result.pp_hidden_states_proxy_tensors,
                                        all_gather_group=self.attn_tp_group,
                                    )
                            pp_outputs = next_outputs
                    self._pp_timing.mark("decode", "control")
                    self._pp_summary()
                    step += 1
                idle = (
                    not self._pp_prefills
                    and not any(self._pp_pending_merges)
                    and not self.waiting_queue
                    and all(
                        not mb.reqs or all(r.finished() for r in mb.reqs)
                        for mb in self.running_mbs
                    )
                    and not any(in_flight)
                )
                if idle:
                    if self._pp_receive_pool is not None:
                        self._pp_receive_pool.assert_idle()
                    with torch.cuda.stream(self._pp_decode_stream):
                        self.check_memory()
                        self.check_tree_cache()
                    if not self._pp_idle_logged:
                        logger.info(
                            "MuxWise-PP idle self-check PASS stage=%s free_kv=%s evictable=%s free_reqs=%s prefill_slices=%s decode_steps=%s cached_tokens=%s overlap_launches=%s free_receive_slots=%s",
                            self.pp_rank,
                            self.token_to_kv_pool_allocator.available_size(),
                            self.tree_cache.evictable_size(),
                            self.req_to_token_pool.available_size(),
                            self._pp_prefill_steps,
                            self._pp_decode_steps,
                            self._pp_cached_tokens,
                            self._pp_overlap_launches,
                            (
                                len(self._pp_receive_pool.free)
                                if self._pp_receive_pool
                                else 0
                            ),
                        )
                    self._pp_idle_logged = True
                    self._pp_timing.mark("decode", "empty")
                    if self.pp_rank == 0:
                        time.sleep(0.001)
                else:
                    self._pp_idle_logged = False
        finally:
            if transport:
                transport.close()
            self.prefill_mailbox.outbox.put(None)
            self.prefill_mailbox.transfers.put(None)
