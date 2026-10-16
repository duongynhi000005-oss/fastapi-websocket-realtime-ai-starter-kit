#!/usr/bin/env python3
"""
Unit test suite (`test_starter_kit.py`) for the Production FastAPI + WebSocket
Real-Time Audio/LLM Streaming & Fault-Isolated Task Supervisor Starter Kit.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import unittest

from dedup_webhook_fanout import FanoutDispatcher, WebhookDeduplicator
from supervisor import SupervisionPolicy, TaskSupervisor
from ws_gateway import WebSocketSessionGateway


class TestStarterKit(unittest.IsolatedAsyncioTestCase):
    async def test_supervisor_retry_and_success(self) -> None:
        sup = TaskSupervisor(max_concurrency=4)
        attempts = 0

        async def flaky_job() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient network glitch")
            return "recovered"

        task = sup.spawn(
            "flaky-1",
            flaky_job,
            policy=SupervisionPolicy(max_retries=3, base_delay_sec=0.01, timeout_sec=1.0),
        )
        result = await task
        self.assertEqual(result, "recovered")
        self.assertEqual(attempts, 3)
        self.assertEqual(sup.completed_count, 1)
        self.assertEqual(sup.failed_count, 0)
        self.assertEqual(sup.active_count, 0)
        await sup.shutdown()

    async def test_supervisor_dead_letter_capture(self) -> None:
        sup = TaskSupervisor(max_concurrency=2)

        async def always_fails() -> None:
            raise ValueError("invalid audio codec frame")

        task = sup.spawn(
            "bad-codec",
            always_fails,
            policy=SupervisionPolicy(max_retries=1, base_delay_sec=0.01, timeout_sec=0.5),
        )
        res = await task
        self.assertIsNone(res)
        self.assertEqual(sup.failed_count, 1)
        self.assertEqual(len(sup.dead_letters), 1)
        self.assertEqual(sup.dead_letters[0].task_name, "bad-codec")
        self.assertEqual(sup.dead_letters[0].error_type, "ValueError")
        self.assertEqual(sup.dead_letters[0].attempts, 2)
        await sup.shutdown()

    async def test_ws_gateway_drop_oldest_backpressure(self) -> None:
        gw = WebSocketSessionGateway(max_queue_size=3, backpressure_mode="drop_oldest")
        gw.register_session("sess-1")

        for i in range(5):
            accepted = gw.push_frame("sess-1", f"pcm_chunk_{i}")
            self.assertTrue(accepted)

        metrics = gw.get_metrics("sess-1")
        assert metrics is not None
        self.assertEqual(metrics.enqueued_frames, 5)
        self.assertEqual(metrics.dropped_frames, 2)

        delivered: list[str] = []

        async def fake_send(frame: str) -> None:
            delivered.append(frame)

        count = await gw.drain_to_sender("sess-1", fake_send)
        self.assertEqual(count, 3)
        self.assertEqual(delivered, ["pcm_chunk_2", "pcm_chunk_3", "pcm_chunk_4"])
        gw.unregister_session("sess-1")
        self.assertEqual(gw.active_sessions, [])

    async def test_webhook_dedup_and_isolated_fanout(self) -> None:
        dedup = WebhookDeduplicator(max_entries=128, ttl_sec=60.0)
        dispatcher = FanoutDispatcher(deduplicator=dedup, per_target_timeout_sec=0.1)

        raw = b'{"session":"s1","transcript":"hello world"}'
        secret = "top_secret_key"
        sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
        self.assertTrue(dedup.verify_hmac_sha256(secret, raw, sig))

        received: list[str] = []

        async def fast_hook(payload: dict) -> None:
            received.append(payload["transcript"])

        async def slow_hook(_payload: dict) -> None:
            await asyncio.sleep(0.5)

        payload = {"session": "s1", "transcript": "hello world"}
        res1 = await dispatcher.dispatch(payload, {"crm": fast_hook, "slow_analytics": slow_hook})
        self.assertFalse(res1.deduplicated)
        self.assertEqual(res1.succeeded_targets, ["crm"])
        self.assertIn("slow_analytics", res1.failed_targets)
        self.assertEqual(received, ["hello world"])

        # Duplicate replay should be suppressed
        res2 = await dispatcher.dispatch(payload, {"crm": fast_hook})
        self.assertTrue(res2.deduplicated)
        self.assertEqual(len(received), 1)


if __name__ == "__main__":
    unittest.main()
