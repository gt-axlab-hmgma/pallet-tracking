import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any


class EventBus:
    """Fans events out to every connected browser. ``publish`` is safe to call from any thread."""

    def __init__(self) -> None:
        self._subs: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subs[queue] = asyncio.get_running_loop()
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.pop(queue, None)

    def publish(self, event: dict[str, Any]) -> None:
        for queue, loop in list(self._subs.items()):
            loop.call_soon_threadsafe(queue.put_nowait, event)


async def sse_lines(
    bus: EventBus,
    disconnected: Callable[[], Awaitable[bool]],
    keepalive_s: float = 15.0,
) -> AsyncIterator[str]:
    """Server-Sent Events frames: a hello, one frame per event, and periodic keepalives."""
    queue = bus.subscribe()
    try:
        yield ": connected\n\n"
        while not await disconnected():
            try:
                event = await asyncio.wait_for(queue.get(), keepalive_s)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield f"event: zone_event\ndata: {json.dumps(event)}\n\n"
    finally:
        bus.unsubscribe(queue)
