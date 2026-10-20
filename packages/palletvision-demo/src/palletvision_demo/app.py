import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from palletvision_demo.live.runner import LiveRunner
from palletvision_demo.live.state import (
    DemoState,
    ObjectInputError,
    Result,
    UnknownObjectError,
    ZoneInputError,
)
from palletvision_demo.live.stream import sse_lines
from palletvision_demo.settings import Settings
from palletvision_infer.backends import OnnxRuntimeBackend
from palletvision_infer.detections import Detector

BOUNDARY = "frame"


class ZonesBody(BaseModel):
    zones: list[dict[str, Any]]


class NewObject(BaseModel):
    object_type: str
    box: list[float]


class ObjectUpdate(BaseModel):
    object_type: str | None = None
    box: list[float] | None = None


def _result(result: Result) -> dict[str, Any]:
    return {
        "events": result.events,
        "sink_error": result.sink_error,
        "config_error": result.config_error,
    }


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """With ``settings.weights`` the source video is tracked live; ``detector`` replaces the ONNX
    model it would load (for tests)."""
    settings = settings or Settings()
    state = DemoState(settings)
    runner: LiveRunner | None = None
    if settings.weights is not None:
        if settings.source is None:
            raise ValueError("live tracking needs --source, the video to run on")
        detector = detector or OnnxRuntimeBackend(settings.weights, conf=settings.conf)
        runner = LiveRunner(state, detector, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runner:
            runner.start()
        yield
        if runner:
            await asyncio.to_thread(runner.stop)

    app = FastAPI(
        title="palletvision demo",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.demo = state
    app.state.runner = runner

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "camera_id": settings.camera_id, "device_id": settings.device_id}

    @app.get("/api/frame")
    def frame() -> Response:
        return Response(
            state.frame_jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
        )

    @app.get("/api/live/stream")
    async def live_stream(request: Request) -> StreamingResponse:
        """The video as motion JPEG, one part per processed frame, for an <img> to play."""

        async def parts() -> AsyncIterator[bytes]:
            seq = 0
            while not await request.is_disconnected():
                seq, jpeg = await asyncio.to_thread(state.wait_frame, seq, 1.0)
                if jpeg is None:
                    continue
                head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}"
                yield head.encode() + b"\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(
            parts(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/state")
    def get_state() -> dict[str, object]:
        return state.snapshot()

    @app.put("/api/zones")
    def put_zones(body: ZonesBody) -> dict[str, object]:
        try:
            result = state.replace_zones(body.zones)
        except ZoneInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**state.snapshot(), **_result(result)}

    @app.post("/api/objects")
    def add_object(body: NewObject) -> dict[str, Any]:
        try:
            obj, result = state.add_object(body.object_type, body.box)
        except ObjectInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"object": state.object_dict(obj), **_result(result)}

    @app.put("/api/objects/{object_id}")
    def update_object(object_id: int, body: ObjectUpdate) -> dict[str, Any]:
        try:
            obj, result = state.update_object(object_id, body.box, body.object_type)
        except ObjectInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnknownObjectError as exc:
            raise HTTPException(status_code=404, detail="unknown object") from exc
        return {"object": state.object_dict(obj), **_result(result)}

    @app.delete("/api/objects/{object_id}")
    def delete_object(object_id: int) -> dict[str, Any]:
        try:
            result = state.delete_object(object_id)
        except UnknownObjectError as exc:
            raise HTTPException(status_code=404, detail="unknown object") from exc
        return _result(result)

    @app.get("/api/events")
    def recent_events(limit: int = 100) -> dict[str, Any]:
        return {"events": state.recent_events(min(limit, 500))}

    @app.get("/api/events/stream")
    async def event_stream(request: Request) -> StreamingResponse:
        return StreamingResponse(
            sse_lines(state.bus, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
