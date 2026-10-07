import argparse
import random
import signal
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from palletvision_events.db import connection_url
from palletvision_events.outbox import Outbox
from palletvision_events.sinks import EventSink

SinkFactory = Callable[[], EventSink]


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.2) -> float:
    """Exponential backoff with a cap and optional +/- jitter (fraction of the delay)."""
    delay: float = min(cap, base * 2**attempt)
    if jitter:
        delay *= 1 + random.uniform(-jitter, jitter)
    return delay


class Syncer:
    """Moves events from the local outbox to a sink, at least once and in order.

    A batch is only marked sent after the sink accepted it. If the sink fails the batch stays
    in the outbox and is retried. The sink must be idempotent (PostgresSink is), so a retry
    after a lost acknowledgement cannot create duplicates.
    """

    def __init__(
        self,
        outbox: Outbox,
        sink_factory: SinkFactory,
        batch_size: int = 200,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.outbox = outbox
        self.batch_size = batch_size
        self.failures = 0
        self._factory = sink_factory
        self._sink: EventSink | None = None
        self._sleep = sleep

    def sync_once(self) -> int:
        """Send one batch. Returns how many events were sent; raises if the sink failed."""
        rows = self.outbox.pending(self.batch_size)
        if not rows:
            return 0
        try:
            if self._sink is None:
                self._sink = self._factory()
            self._sink.write([event for _, event in rows])
        except Exception:
            self._discard_sink()
            raise
        self.outbox.mark_sent([row_id for row_id, _ in rows])
        return len(rows)

    def drain(self) -> int:
        """Send batches until the outbox is empty. Returns the total sent."""
        total = 0
        while sent := self.sync_once():
            total += sent
        return total

    def run(self, interval: float, stop: threading.Event, purge_after_s: float = 86400) -> None:
        attempt = 0
        last_purge = time.monotonic()
        while not stop.is_set():
            try:
                self.drain()
                attempt = 0
                delay = interval
            except Exception as exc:
                self.failures += 1
                delay = backoff_delay(attempt)
                attempt += 1
                print(f"pv-sync: {type(exc).__name__}: {exc}; retrying in {delay:.1f}s", flush=True)
            if time.monotonic() - last_purge > 60:
                self.outbox.purge_sent(purge_after_s)
                last_purge = time.monotonic()
            if stop.is_set():
                break
            if delay:
                (self._sleep or stop.wait)(delay)
        self._discard_sink()

    def _discard_sink(self) -> None:
        if self._sink is not None:
            try:
                self._sink.close()
            finally:
                self._sink = None


def main(argv: Sequence[str] | None = None) -> None:
    from palletvision_events.sinks_postgres import PostgresSink

    p = argparse.ArgumentParser(
        prog="pv-sync", description="Send events from the local outbox to Postgres."
    )
    p.add_argument("--outbox", type=Path, default=Path("runs/outbox.db"), help="outbox SQLite file")
    p.add_argument("--db-url", default=None, help="Postgres URL (default: from PV_DB_* / .env)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between polls when idle")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--once", action="store_true", help="drain the outbox once and exit")
    args = p.parse_args(argv)

    url = args.db_url or connection_url()
    syncer = Syncer(Outbox(args.outbox), lambda: PostgresSink(url), args.batch_size)
    if args.once:
        print(f"sent {syncer.drain()} events")
        return
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"pv-sync: draining {args.outbox} -> {url.split('@')[-1]} (Ctrl+C to stop)", flush=True)
    syncer.run(args.interval, stop)
