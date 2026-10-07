from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from palletvision_demo.settings import Settings
from palletvision_demo.state import (
    DemoState,
    ObjectInputError,
    Result,
    UnknownObjectError,
    ZoneInputError,
)


class ZonesBody(BaseModel):
    zones: list[dict[str, Any]]


class NewObject(BaseModel):
    object_type: str
    box: list[float]


class ObjectUpdate(BaseModel):
    object_type: str | None = None
    box: list[float] | None = None


def _result(result: Result) -> dict[str, Any]:
    return {"events": result.events, "sink_error": result.sink_error}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = DemoState(settings)
    app = FastAPI(title="palletvision demo", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.demo = state

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "camera_id": settings.camera_id, "device_id": settings.device_id}

    @app.get("/api/frame")
    def frame() -> Response:
        return Response(
            state.frame_jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
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

    return app
