#!/usr/bin/env python3
"""
Real-Time WebSocket Audio/LLM Streaming Gateway (`ws_gateway.py`)
Part of the Production FastAPI + WebSocket Real-Time Audio/LLM Streaming Starter Kit.

Provides bounded per-connection outbound queues, deterministic slow-consumer backpressure
(`drop_oldest` or `reject`), and clean session lifecycle management.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional

BackpressureMode = Literal["drop_oldest", "reject"]


@dataclass
class SessionMetrics:
    session_id: str
    enqueued_frames: int = 0
    delivered_frames: int = 0
    dropped_frames: int = 0
    last_activity_ts: float = field(default_factory=time.time)


class WebSocketSessionGateway:
    """
    Manages real-time bidirectional streaming sessions (PCM16 audio frames / LLM tokens)
    without allowing slow or stalled clients to exhaust server RAM.
    """

    def __init__(
        self,
        max_queue_size: int = 64,
        backpressure_mode: BackpressureMode = "drop_oldest",
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        self._max_queue_size = max_queue_size
        self._backpressure_mode: BackpressureMode = backpressure_mode
        self._queues: Dict[str, asyncio.Queue[Any]] = {}
        self._metrics: Dict[str, SessionMetrics] = {}

    def register_session(self, session_id: str) -> SessionMetrics:
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=self._max_queue_size)
            self._metrics[session_id] = SessionMetrics(session_id=session_id)
        return self._metrics[session_id]

    def unregister_session(self, session_id: str) -> Optional[SessionMetrics]:
        q = self._queues.pop(session_id, None)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except asyncio.QueueEmpty:
                    break
        return self._metrics.pop(session_id, None)

    def push_frame(self, session_id: str, frame: Any) -> bool:
        """
        Non-blocking frame push with deterministic backpressure handling.
        Returns True if the frame was accepted into the session queue, False if rejected.
        """
        if session_id not in self._queues:
            raise KeyError(f"Unknown session_id: {session_id}")
        q = self._queues[session_id]
        m = self._metrics[session_id]
        m.last_activity_ts = time.time()

        if q.full():
            if self._backpressure_mode == "reject":
                m.dropped_frames += 1
                return False
            # drop_oldest: evict oldest frame so real-time audio/token stream stays low-latency
            try:
                q.get_nowait()
                q.task_done()
                m.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass

        q.put_nowait(frame)
        m.enqueued_frames += 1
        return True

    async def drain_to_sender(
        self,
        session_id: str,
        send_fn: Callable[[Any], Coroutine[Any, Any, None]],
        max_frames: Optional[int] = None,
    ) -> int:
        if session_id not in self._queues:
            return 0
        q = self._queues[session_id]
        m = self._metrics[session_id]
        sent = 0
        while not q.empty():
            if max_frames is not None and sent >= max_frames:
                break
            frame = await q.get()
            try:
                await send_fn(frame)
                m.delivered_frames += 1
                sent += 1
            finally:
                q.task_done()
        return sent

    def get_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        return self._metrics.get(session_id)

    @property
    def active_sessions(self) -> List[str]:
        return list(self._queues.keys())
