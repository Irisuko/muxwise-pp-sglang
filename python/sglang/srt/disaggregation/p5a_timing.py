"""Opt-in same-host monotonic timestamps; no CUDA sync or scheduling changes."""
import logging
import os
import time

_enabled = os.getenv("P5A_KV_TIMING") == "1"
_logger = logging.getLogger(__name__)


def kv_stamp(event, req, start=0, end=0, last=False):
    if _enabled:
        _logger.info(
            "P5A_KV event=%s ns=%d room=%s tokens=%d start=%d end=%d last=%d",
            event, time.monotonic_ns(), req.bootstrap_room,
            len(req.origin_input_ids), start, end, last,
        )
