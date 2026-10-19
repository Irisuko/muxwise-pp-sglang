"""MuxWise PP: ordered decode controls and an independent asynchronous prefill lane.

Only the scheduler thread touches request/cache state. Daemon threads
transport CPU envelopes/readiness; another launches NCCL on the selected prefill stream
(even ncclGroupEnd can block the host on first peer connection). In particular, an
unmatched receive is never followed by a stream synchronize/partition change.
"""

import logging
import queue
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
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.multiplex.pdmux_context import set_current_stream_idx

logger = logging.getLogger(__name__)


class ControlKind(str, Enum):
    ADMIT = "admit"
    MERGE = "merge"
    RETRACT = "retract"
    ABORT = "abort"
    MIGRATE = "migrate"  # P2: ordered rids, source_mb, target_mb; not emitted in P1.


@dataclass
class PrefillPlan:
    batch_id: int
    # Include unsuccessful probes: match_prefix can split radix nodes.
    probes: list[tuple[str, int, int]]  # rid, fill length, prefix length
    rids: list[str]


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


class StaticMuxPPPolicy:
    """P2 replaces this object; only admission/merge hooks run on stage zero."""

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
            self.inbox.put(dict(kind="ready", batch_id=int(bid.item())))

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
                "MuxWise-PP P1 supports text/token generation without "
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
        assert self.tp_size == 1, "MuxWise-PP P1 supports TP=1"
        assert (
            self.model_config.hf_config.model_type == "llama"
        ), "MuxWise-PP P1 supports Llama"
        assert (
            type(self.tree_cache) is RadixCache
        ), "MuxWise-PP requires the Python RadixCache"
        if self.server_args.pdmux_pp_radix_eviction == "fifo":
            self.tree_cache.eviction_strategy = DeterministicRadixFIFO()
        self._pp_eviction = OrderedRadixEviction(self.tree_cache, self.pp_rank == 0)
        self.mux_pp_policy = StaticMuxPPPolicy()
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
        self._pp_stream_idx = 0
        self._pp_idle_logged = False
        self._pp_prefill_steps = 0
        self._pp_decode_steps = 0
        self._pp_cached_tokens = 0
        self._pp_overlap_launches = 0
        self._pp_select_streams(0)
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

    def _pp_select_streams(self, idx):
        self._pp_stream_idx = idx
        set_current_stream_idx(idx)
        self._pp_prefill_stream, self._pp_decode_stream = self.stream_groups[idx]
        self.tp_worker.model_runner.update_decode_attn_backend(idx)

    def _pp_adjust_streams(self, decode_bs):
        if not decode_bs:
            decode_bs = max(mb.batch_size() for mb in self.running_mbs)
        idx = self.mux_pp_policy.partition(
            self.pdmux_config,
            len(self.stream_groups),
            decode_bs,
            any(w.state != "done" for w in self._pp_prefills.values()),
        )
        if idx == self._pp_stream_idx:
            return
        # An unmatched NCCL receive/send must never prevent decode progress.
        for work in self._pp_prefills.values():
            if work.launched is not None and not work.launched.is_set():
                return
            if work.event is not None and not work.event.query():
                return
        self._pp_prefill_stream.synchronize()
        self._pp_decode_stream.synchronize()
        self._pp_select_streams(idx)
        logger.debug(
            "MuxWise-PP partition=%s decode_bs=%s SMs=%s",
            idx,
            decode_bs,
            self.sm_counts[idx],
        )

    def _pp_make_plan(self):
        if len(self._pp_prefills) >= self._pp_max_prefills or not self.waiting_queue:
            return None
        budget = self.mux_pp_policy.admission_budget(self)
        if budget <= 0:
            return None
        # Reserve the entire remaining output, across ALL microbatches, plus
        # one pipeline turn of safety. P2 may replace this conservative rule.
        active = [r for mb in self.running_mbs for r in mb.reqs if not r.finished()]
        pending = [r for w in self._pp_prefills.values() for r in w.batch.reqs]
        active += pending  # input KV already allocated; reserve future output too
        reserve = sum(
            max(0, r.sampling_params.max_new_tokens - len(r.output_ids)) + self.pp_size
            for r in active
        )
        target = self.mux_pp_policy.merge_target(self.running_mbs)
        max_bs = self.server_args.max_micro_batch_size or max(
            1, self.max_running_requests // self.pp_size
        )
        slots = min(
            self.req_to_token_pool.available_size(),
            max_bs
            - sum(not r.finished() for r in self.running_mbs[target].reqs)
            - len(pending),
        )
        selected, probes = [], []
        input_tokens = 0
        for req in self.waiting_queue:
            if len(selected) >= slots:
                break
            req.init_next_round_input(self.tree_cache)
            probes.append((req.rid, len(req.fill_ids), len(req.prefix_indices)))
            self.tree_cache.inc_lock_ref(req.last_node)
            needed = (
                req.extend_input_len + req.sampling_params.max_new_tokens + self.pp_size
            )
            available = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
            if reserve + needed >= available or (
                selected and input_tokens + req.extend_input_len > budget
            ):
                self.tree_cache.dec_lock_ref(req.last_node)
                break
            selected.append(req)
            reserve += needed
            input_tokens += req.extend_input_len
        plan = PrefillPlan(self._pp_batch_id, probes, [r.rid for r in selected])
        if selected:
            self._pp_batch_id += 1
            self._pp_build_prefill(plan, selected)
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
        while (envelope := self.prefill_mailbox.poll()) is not None:
            bid = envelope["batch_id"]
            if envelope["kind"] == "ready":
                self.prefill_mailbox.ready.add(bid)
                continue
            if self.pp_rank == 0:
                assert envelope["kind"] == "result" and bid in self._pp_prefills
                self._pp_merge_ready.append(
                    dict(
                        batch_id=bid,
                        target_mb=self.mux_pp_policy.merge_target(self.running_mbs),
                        next_token_ids=envelope["tokens"],
                    )
                )
            else:
                assert bid not in self._pp_envelopes
                self._pp_envelopes[bid] = envelope
        # Local FIFO: a sent batch remains allocated until ordered MERGE, but
        # no longer owns the compute slot. Receive only the FIFO head to bound
        # activation memory and avoid queueing unmatched NCCL on the stream.
        work = next((w for w in self._pp_prefills.values() if w.state != "done"), None)
        self.split_prefill_batch = work.batch if work else None
        if work is None:
            return
        if work.state == "waiting":
            envelope = self._pp_envelopes.pop(work.plan.batch_id, None)
            if envelope is None:
                return
            assert envelope["plan"] == work.plan
            tensors = {
                k: torch.empty(shape, dtype=dtype, device=self.device)
                for k, shape, dtype in envelope["tensors"]
            }
            work.proxy = PPProxyTensors(tensors)
            work.launched, work.event = self.prefill_mailbox.transfer(
                False, tensors, self._pp_prefill_stream
            )
            work.state = "receiving"
            self.prefill_mailbox.outbox.put(
                dict(kind="ready", batch_id=work.plan.batch_id)
            )
        if work.state == "offering":
            if work.plan.batch_id not in self.prefill_mailbox.ready:
                return
            self.prefill_mailbox.ready.remove(work.plan.batch_id)
            work.launched, work.event = self.prefill_mailbox.transfer(
                True, work.proxy.tensors, self._pp_prefill_stream
            )
            work.state = "sending"
        if work.launched is not None:
            if not work.launched.is_set():
                return
            work.launched = None
        if work.event is not None and not work.event.query():
            return
        if work.state == "sending":
            work.state, work.proxy = "done", None
            work.batch.split_forward_batch = None
            self.split_prefill_batch = None
            return
        if work.state == "receiving":
            work.state = "compute"
        batch = work.batch
        end = self.tp_worker.model_runner.model.end_layer
        if batch.split_index < end:
            count = self.mux_pp_policy.slice_layers(
                self.pdmux_config, batch.extend_num_tokens, end - batch.split_index
            )
            batch.split_forward_count = count
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
            self.prefill_mailbox.outbox.put(
                dict(
                    kind="result",
                    batch_id=work.plan.batch_id,
                    tokens=work.tokens.tolist(),
                )
            )
            work.state = "done"
            work.batch.split_forward_batch = None
            self.split_prefill_batch = None
        else:
            tensors = work.proxy.tensors
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

    def _pp_merge(self, merge, mb_id):
        work = self._pp_prefills[merge["batch_id"]]
        assert merge["target_mb"] == mb_id
        batch = work.batch
        assert batch.split_prefill_finished
        self._pp_decode_stream.wait_event(work.event)
        batch.output_ids = torch.tensor(
            merge["next_token_ids"], device=self.device, dtype=torch.int64
        )
        self.process_batch_result(batch, self._pp_result(batch.output_ids))
        batch.split_forward_batch = (
            None  # Release hidden/residual and attention metadata.
        )
        batch.filter_batch()
        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)
        self.running_mbs[mb_id] = self.running_batch
        del self._pp_prefills[merge["batch_id"]]
        if self.split_prefill_batch is batch:
            self.split_prefill_batch = None
        if self.pp_rank == 0:
            assert self._pp_merge_ready.popleft()["batch_id"] == merge["batch_id"]

    def _pp_abort_ids(self, requests):
        aborts = [r for r in requests if isinstance(r, AbortReq)]
        reqs = self.waiting_queue + [r for mb in self.running_mbs for r in mb.reqs]
        reqs += [r for w in self._pp_prefills.values() for r in w.batch.reqs]
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
        step = 0
        try:
            while True:
                for mb_id in range(self.pp_size):
                    self.running_batch = self.running_mbs[mb_id]
                    self.cur_batch = None
                    self.last_batch = None
                    self._pp_adjust_streams(self.running_batch.batch_size())
                    with torch.cuda.stream(self._pp_decode_stream):
                        set_pdmux_status(False)
                        if self.pp_rank == 0:
                            requests = self.recv_requests()
                            packet = PPControl(step, mb_id, requests=requests)
                        else:
                            obj = [None]
                            dist.recv_object_list(
                                obj,
                                src=self.pp_group.ranks[self.pp_rank - 1],
                                group=self.world_group.cpu_group,
                            )
                            packet = obj[0]
                            assert (packet.step, packet.mb_id) == (step, mb_id)
                        self._pp_eviction.begin(packet.evictions)
                        # Abort decisions include all microbatches and in-flight prefill.
                        self.process_input_requests(
                            [r for r in packet.requests if not isinstance(r, AbortReq)]
                        )
                        if self.pp_rank == 0:
                            packet.abort = self._pp_abort_ids(packet.requests)
                        self._pp_apply_aborts(packet.abort)
                        if self.pp_rank == 0 and self._pp_merge_ready:
                            if self._pp_merge_ready[0]["target_mb"] == mb_id:
                                packet.merge = self._pp_merge_ready[0]
                        if packet.merge:
                            self._pp_merge(packet.merge, mb_id)
                        if self.pp_rank == 0:
                            packet.plan = self._pp_make_plan()
                        elif packet.plan:
                            self._pp_apply_plan(packet.plan)
                        if packet.migrations:
                            raise NotImplementedError(
                                "P2 microbatch migration is not enabled"
                            )
                        self.running_batch.filter_batch()
                        # Only stage zero decides retraction. Full output reservation
                        # normally makes this branch unnecessary; keep an explicit command.
                        if self.pp_rank == 0 and not self.running_batch.is_empty():
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
                        if not self.running_batch.is_empty():
                            self.running_batch.prepare_for_decode()
                            self.cur_batch = self.running_batch
                        self.running_mbs[mb_id] = self.running_batch
                        in_flight[mb_id] = self.cur_batch
                        self._pp_eviction.end()
                    with torch.cuda.stream(self._pp_prefill_stream):
                        set_pdmux_status(True)
                        previous_slices = self._pp_prefill_steps
                        self._pp_progress_prefill()
                        if self.cur_batch and self._pp_prefill_steps > previous_slices:
                            self._pp_overlap_launches += 1
                    with torch.cuda.stream(self._pp_decode_stream):
                        set_pdmux_status(False)
                        result = None
                        if self.cur_batch:
                            result = self.run_batch(self.cur_batch)
                            self._pp_decode_steps += 1
                        # This is the upstream PP ring: retire the next microbatch,
                        # relay the previous token result, then send controls/proxy.
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
                            next_outputs = self.pp_group.recv_tensor_dict(
                                all_gather_group=self.attn_tp_group
                            )
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
                    step += 1
                idle = (
                    not self._pp_prefills
                    and not self.waiting_queue
                    and all(
                        not mb.reqs or all(r.finished() for r in mb.reqs)
                        for mb in self.running_mbs
                    )
                    and not any(in_flight)
                )
                if idle:
                    with torch.cuda.stream(self._pp_decode_stream):
                        self.check_memory()
                        self.check_tree_cache()
                    if not self._pp_idle_logged:
                        logger.info(
                            "MuxWise-PP idle self-check PASS stage=%s free_kv=%s evictable=%s free_reqs=%s prefill_slices=%s decode_steps=%s cached_tokens=%s overlap_launches=%s",
                            self.pp_rank,
                            self.token_to_kv_pool_allocator.available_size(),
                            self.tree_cache.evictable_size(),
                            self.req_to_token_pool.available_size(),
                            self._pp_prefill_steps,
                            self._pp_decode_steps,
                            self._pp_cached_tokens,
                            self._pp_overlap_launches,
                        )
                    self._pp_idle_logged = True
                    if self.pp_rank == 0:
                        time.sleep(0.001)
                else:
                    self._pp_idle_logged = False
        finally:
            self.prefill_mailbox.outbox.put(None)
            self.prefill_mailbox.transfers.put(None)
