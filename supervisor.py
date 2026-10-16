#!/usr/bin/env python3
"""
Fault-Isolated Asyncio Task Supervisor (`supervisor.py`)
Part of the Production FastAPI + WebSocket Real-Time Audio/LLM Streaming Starter Kit.

Prevents silent background task exceptions, unbounded task spawning, and memory leaks
in long-running FastAPI / WebSocket servers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(frozen=True)
class SupervisionPolicy:
    max_retries: int = 2
    base_delay_sec: float = 0.05
    max_delay_sec: float = 1.0
    timeout_sec: Optional[float] = 5.0


@dataclass(frozen=True)
class DeadLetterEvent:
    task_name: str
    error_type: str
    error_message: str
    attempts: int
    timestamp: float


class TaskSupervisor:
    """
    Supervises background asyncio tasks with:
    - Bounded concurrency semaphore
    - Automatic task registry cleanup (zero orphaned task references)
    - Configurable retry + timeout policy per task
    - Bounded dead-letter ring buffer for observability
    """

    def __init__(
        self,
        max_concurrency: int = 64,
        max_dead_letters: int = 256,
        default_policy: Optional[SupervisionPolicy] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._default_policy = default_policy or SupervisionPolicy()
        self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._dead_letters: List[DeadLetterEvent] = []
        self._max_dead_letters = max_dead_letters
        self._closed = False
        self._completed_count = 0
        self._failed_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    @property
    def completed_count(self) -> int:
        return self._completed_count

    @property
    def failed_count(self) -> int:
        return self._failed_count

    @property
    def dead_letters(self) -> List[DeadLetterEvent]:
        return list(self._dead_letters)

    def spawn(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        policy: Optional[SupervisionPolicy] = None,
        deduplicate_by_name: bool = False,
    ) -> asyncio.Task[Any]:
        if self._closed:
            raise RuntimeError("TaskSupervisor is closed and cannot accept new tasks")

        if deduplicate_by_name and name in self._active_tasks:
            existing = self._active_tasks[name]
            if not existing.done():
                return existing

        effective_policy = policy or self._default_policy
        task = asyncio.create_task(
            self._run_supervised(name, coro_factory, effective_policy),
            name=name,
        )
        self._active_tasks[name] = task
        task.add_done_callback(lambda t, k=name: self._on_task_done(k, t))
        return task

    def _on_task_done(self, key: str, task: asyncio.Task[Any]) -> None:
        if self._active_tasks.get(key) is task:
            self._active_tasks.pop(key, None)

    async def _run_supervised(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        policy: SupervisionPolicy,
    ) -> Any:
        attempts = 0
        max_attempts = max(1, policy.max_retries + 1)
        last_exc: Optional[BaseException] = None

        while attempts < max_attempts:
            attempts += 1
            try:
                async with self._semaphore:
                    coro = coro_factory()
                    if policy.timeout_sec is not None:
                        result = await asyncio.wait_for(coro, timeout=policy.timeout_sec)
                    else:
                        result = await coro
                self._completed_count += 1
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempts >= max_attempts:
                    break
                delay = min(
                    policy.max_delay_sec,
                    policy.base_delay_sec * (2 ** (attempts - 1)),
                )
                await asyncio.sleep(delay)

        self._failed_count += 1
        if last_exc is not None:
            event = DeadLetterEvent(
                task_name=name,
                error_type=type(last_exc).__name__,
                error_message=str(last_exc),
                attempts=attempts,
                timestamp=time.time(),
            )
            self._dead_letters.append(event)
            if len(self._dead_letters) > self._max_dead_letters:
                self._dead_letters.pop(0)
        return None

    async def shutdown(self, drain_timeout: float = 2.0) -> None:
        self._closed = True
        tasks = list(self._active_tasks.values())
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=drain_timeout)
        for p in pending:
            p.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._active_tasks.clear()
