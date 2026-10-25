import cv2
import numpy as np
from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings

PAINT = tuple(
    int(v) for v in cv2.cvtColor(np.array([[[27, 90, 190]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
)


def floor_with_one_bay() -> np.ndarray:
    img = np.full((720, 1280, 3), (128, 130, 126), np.uint8)
    cv2.line(img, (300, 600), (700, 615), PAINT, 8)  # the base line
    cv2.line(img, (300, 600), (380, 280), PAINT, 8)  # left wall
    cv2.line(img, (700, 615), (690, 280), PAINT, 8)  # right wall
    return img


def test_before_any_source_there_is_no_frame_to_propose_from() -> None:
    res = TestClient(create_app(Settings(camera_id="camA"))).post("/api/zones/auto")
    assert res.status_code == 422
    assert "first frame" in res.json()["detail"]


def test_the_bays_on_the_current_frame_come_back_as_polygons() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())
    res = TestClient(app).post("/api/zones/auto")
    assert res.status_code == 200
    body = res.json()
    assert len(body["polygons"]) == 1
    xs = [x for x, _ in body["polygons"][0]]
    assert 290 <= min(xs) <= 310 and 690 <= max(xs) <= 710
    assert body["frame_version"] == app.state.demo.frame_version


def test_proposing_changes_no_zone() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())
    client = TestClient(app)
    before = client.get("/api/state").json()
    client.post("/api/zones/auto")
    after = client.get("/api/state").json()
    assert after["zones"] == before["zones"] == []
    assert after["version"] == before["version"]


def test_a_frame_with_no_paint_gives_an_empty_list_not_an_error() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(np.full((720, 1280, 3), (128, 130, 126), np.uint8))
    res = TestClient(app).post("/api/zones/auto")
    assert res.status_code == 200
    assert res.json()["polygons"] == []


def test_a_new_frame_gets_a_new_version_so_an_old_answer_can_be_told_apart() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())
    first = TestClient(app).post("/api/zones/auto").json()["frame_version"]
    app.state.demo.set_reference(floor_with_one_bay())
    assert TestClient(app).post("/api/zones/auto").json()["frame_version"] == first + 1


def test_a_point_on_a_pale_line_gives_the_bay() -> None:
    app = create_app(Settings(camera_id="camA"))
    pale = np.full((720, 1280, 3), (128, 130, 126), np.uint8)
    colour = tuple(
        int(v) for v in cv2.cvtColor(np.array([[[27, 32, 150]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.line(pale, (300, 600), (700, 615), colour, 8)
    cv2.line(pale, (300, 600), (380, 280), colour, 8)
    cv2.line(pale, (700, 615), (690, 280), colour, 8)
    app.state.demo.set_reference(pale)
    client = TestClient(app)
    res = client.post("/api/zones/auto", json={"x": 450, "y": 605})
    assert res.status_code == 200
    assert len(res.json()["polygons"]) == 1


def test_a_point_that_is_not_on_paint_is_refused_with_a_reason() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())
    res = TestClient(app).post("/api/zones/auto", json={"x": 100, "y": 100})
    assert res.status_code == 422
    assert "painted line" in res.json()["detail"]


def test_a_point_outside_the_frame_is_refused() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())
    assert TestClient(app).post("/api/zones/auto", json={"x": 5000, "y": 5}).status_code == 422


def test_the_paint_color_comes_back_so_the_ui_can_draw_in_a_contrasting_one() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(floor_with_one_bay())  # yellow paint, hue 27
    hue = TestClient(app).post("/api/zones/auto").json()["hue"]
    assert 24 <= hue <= 30


def test_no_paint_gives_no_color() -> None:
    app = create_app(Settings(camera_id="camA"))
    app.state.demo.set_reference(np.full((720, 1280, 3), (128, 130, 126), np.uint8))
    assert TestClient(app).post("/api/zones/auto").json()["hue"] is None
