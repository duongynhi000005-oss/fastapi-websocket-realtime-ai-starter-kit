#!/usr/bin/env python3
"""
Idempotent Webhook Fanout & SHA-256 HMAC Deduplicator (`dedup_webhook_fanout.py`)
Part of the Production FastAPI + WebSocket Real-Time Audio/LLM Streaming Starter Kit.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(frozen=True)
class FanoutResult:
    event_id: str
    deduplicated: bool
    succeeded_targets: List[str]
    failed_targets: Dict[str, str]


class WebhookDeduplicator:
    """
    Bounded LRU + TTL deduplicator for incoming webhook payloads or real-time transcript chunks.
    """

    def __init__(self, max_entries: int = 2048, ttl_sec: float = 300.0) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl_sec = ttl_sec
        self._seen: "OrderedDict[str, float]" = OrderedDict()

    @staticmethod
    def compute_fingerprint(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_hmac_sha256(secret: str, raw_body: bytes, signature_hex: str) -> bool:
        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature_hex.lower().removeprefix("sha256="))

    def is_duplicate(self, key: str, now: Optional[float] = None) -> bool:
        ts = now if now is not None else time.time()
        self._evict_expired(ts)
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = ts
        if len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)
        return False

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._ttl_sec
        while self._seen:
            oldest_key, oldest_ts = next(iter(self._seen.items()))
            if oldest_ts < cutoff:
                self._seen.popitem(last=False)
            else:
                break


class FanoutDispatcher:
    """
    Dispatches verified events concurrently across multiple downstream webhook targets
    with per-target timeout isolation so one slow webhook never blocks others.
    """

    def __init__(
        self,
        deduplicator: Optional[WebhookDeduplicator] = None,
        per_target_timeout_sec: float = 3.0,
    ) -> None:
        self._dedup = deduplicator or WebhookDeduplicator()
        self._timeout = per_target_timeout_sec

    async def dispatch(
        self,
        payload: Dict[str, Any],
        targets: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]],
        event_id: Optional[str] = None,
    ) -> FanoutResult:
        eid = event_id or self._dedup.compute_fingerprint(payload)
        if self._dedup.is_duplicate(eid):
            return FanoutResult(
                event_id=eid,
                deduplicated=True,
                succeeded_targets=[],
                failed_targets={},
            )

        succeeded: List[str] = []
        failed: Dict[str, str] = {}

        async def _call_target(name: str, fn: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
            try:
                await asyncio.wait_for(fn(payload), timeout=self._timeout)
                succeeded.append(name)
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {exc}"

        await asyncio.gather(*(_call_target(n, f) for n, f in targets.items()))
        return FanoutResult(
            event_id=eid,
            deduplicated=False,
            succeeded_targets=sorted(succeeded),
            failed_targets=failed,
        )
