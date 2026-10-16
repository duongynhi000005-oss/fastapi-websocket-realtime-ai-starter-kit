# Production FastAPI + WebSocket Real-Time Audio/LLM Streaming & Task Supervisor Starter Kit (v1.0)

A zero-dependency, battle-tested Python concurrency & streaming boilerplate engineered for real-time AI voice agents, WebSocket token/audio gateways, and high-throughput webhook pipelines.

---

## Included Production Modules

1. **`supervisor.py` — Fault-Isolated Asyncio Task Supervisor**
   - Prevents the #1 cause of silent production failures in FastAPI/Asyncio: untracked `asyncio.create_task()` calls whose exceptions are garbage-collected or leak memory.
   - Bounded `asyncio.Semaphore` concurrency, exponential backoff retries, per-task timeouts, task deduplication by key, and a bounded dead-letter ring buffer (`DeadLetterEvent`).
2. **`ws_gateway.py` — Real-Time WebSocket Backpressure & Session Gateway**
   - Bounded per-session outbound queues with deterministic slow-consumer policies (`drop_oldest` for low-latency audio/LLM streaming, or `reject` for strict backpressure).
   - Automatic queue draining upon session disconnect so disconnected clients never leak buffers.
3. **`dedup_webhook_fanout.py` — SHA-256 HMAC Verification & Idempotent Fanout Dispatcher**
   - Constant-time `hmac.compare_digest` signature verification.
   - Bounded LRU + TTL deduplication (`WebhookDeduplicator`) using canonical JSON SHA-256 fingerprints.
   - Concurrent multi-target webhook fanout (`FanoutDispatcher`) with per-target timeout isolation so one slow downstream webhook never stalls the pipeline.
4. **`test_starter_kit.py` — Full Pytest Verification Suite**
   - 100% deterministic async unit tests covering task retries, dead-letter capture, WebSocket frame eviction under slow consumers, and idempotent webhook fanout.

---

## Quickstart & Running Tests

```bash
pytest test_starter_kit.py -v
```
