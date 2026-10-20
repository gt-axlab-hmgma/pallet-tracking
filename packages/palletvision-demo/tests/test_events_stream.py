import asyncio
import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.live.state import DemoState
from palletvision_demo.live.stream import EventBus, sse_lines
from palletvision_demo.settings import Settings
from palletvision_zones.engine import Track

ZONE = {"id": "z", "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]]}


def make_client(tmp_path: Path) -> TestClient:
    c = TestClient(create_app(Settings(camera_id="camA", jsonl_path=tmp_path / "e.jsonl")))
    c.put("/api/zones", json={"zones": [ZONE]})
    return c


def test_events_endpoint_returns_recent_events_oldest_first(tmp_path: Path) -> None:
    c = make_client(tmp_path)
    state: DemoState = c.app.state.demo  # type: ignore[attr-defined]
    state.apply_tracks([Track(1, "pallet_unknown", (50.0, 50.0, 90.0, 90.0))], [])
    state.apply_tracks([Track(1, "pallet_unknown", (500.0, 500.0, 540.0, 540.0))], [])
    events = c.get("/api/events").json()["events"]
    assert [(e["event_type"], e["message"]) for e in events] == [
        ("enter", "object of type pallet_unknown joined zone z"),
        ("exit", "object of type pallet_unknown left zone z"),
    ]
    assert len(c.get("/api/events", params={"limit": 1}).json()["events"]) == 1
    assert c.get("/api/events", params={"limit": 1}).json()["events"][0]["event_type"] == "exit"


def test_bus_delivers_events_published_from_another_thread() -> None:
    async def scenario() -> dict[str, int]:
        bus = EventBus()
        queue = bus.subscribe()
        threading.Thread(target=bus.publish, args=({"n": 1},)).start()
        got = await asyncio.wait_for(queue.get(), 2)
        bus.unsubscribe(queue)
        bus.publish({"n": 2})  # nobody is listening any more; must not raise
        return got

    assert asyncio.run(scenario()) == {"n": 1}


def test_sse_lines_frames_events_sends_keepalives_and_stops() -> None:
    async def scenario() -> list[str]:
        bus = EventBus()
        state = {"calls": 0}

        async def disconnected() -> bool:
            state["calls"] += 1
            return state["calls"] > 3

        out: list[str] = []
        gen = sse_lines(bus, disconnected, keepalive_s=0.05)
        out.append(await anext(gen))  # connected comment
        bus.publish({"event_type": "enter", "zone_id": "z"})
        async for line in gen:
            out.append(line)
        return out

    lines = asyncio.run(scenario())
    assert lines[0].startswith(":")
    data = [ln for ln in lines if ln.startswith("event: zone_event")]
    assert len(data) == 1
    payload = next(ln for ln in lines if "event: zone_event" in ln).split("data: ")[1].strip()
    assert json.loads(payload) == {"event_type": "enter", "zone_id": "z"}
    assert any(ln.startswith(": keepalive") for ln in lines)
