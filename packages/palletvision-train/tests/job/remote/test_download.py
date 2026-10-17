"""Resumable downloads, against a real local server that drops and stalls."""

import http.server
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from palletvision_train.job.remote import download


class Flaky(http.server.BaseHTTPRequestHandler):
    """Serves ``PAYLOAD`` with Range support; the first full download is cut off halfway."""

    PAYLOAD = bytes(range(256)) * 8000  # about 2 MB
    cut_first = True
    seen: ClassVar[list[str]] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.PAYLOAD)))
        self.end_headers()

    def do_GET(self) -> None:
        start = 0
        ranged = self.headers.get("Range")
        if ranged:
            start = int(ranged.removeprefix("bytes=").rstrip("-"))
        self.seen.append(f"range {start}" if ranged else "full")
        body = self.PAYLOAD[start:]
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if Flaky.cut_first and not ranged:
            Flaky.cut_first = False
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.close()  # a dropped connection, with half the body still owed
            return
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[str]:
    Flaky.cut_first, Flaky.seen = True, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin"
    httpd.shutdown()


def test_a_dropped_download_resumes_from_where_it_stopped(tmp_path: Path, server: str) -> None:
    destination = tmp_path / "out" / "file.bin"
    download.download_resumable(server, destination)
    assert destination.read_bytes() == Flaky.PAYLOAD
    assert (
        Flaky.seen[0] == "full"
        and Flaky.seen[1].startswith("range ")
        and Flaky.seen[1] != "range 0"
    )
    assert not destination.with_name("file.bin.part").exists()


def test_a_stalled_connection_times_out_and_resumes(
    tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that goes silent mid-body (without closing) must not hang the download forever."""
    monkeypatch.setattr(download, "TIMEOUT", 1)
    stalled = threading.Event()
    original = Flaky.do_GET

    def stall(self: Flaky) -> None:
        if self.headers.get("Range") or stalled.is_set():
            return original(self)
        stalled.set()
        self.send_response(200)
        self.send_header("Content-Length", str(len(Flaky.PAYLOAD)))
        self.end_headers()
        self.wfile.write(Flaky.PAYLOAD[:1000])
        self.wfile.flush()
        time.sleep(20)  # silent for far longer than the timeout, connection still open

    monkeypatch.setattr(Flaky, "do_GET", stall)
    destination = tmp_path / "file.bin"
    began = time.monotonic()
    download.download_resumable(server, destination, pause=lambda s: None)
    assert destination.read_bytes() == Flaky.PAYLOAD
    assert time.monotonic() - began < 10  # it did not sit out the server's 20 s silence


def test_a_finished_download_is_not_fetched_again(tmp_path: Path, server: str) -> None:
    destination = tmp_path / "file.bin"
    destination.write_bytes(Flaky.PAYLOAD)  # left by a run that was cut off while unpacking it
    download.download_resumable(server, destination)
    assert Flaky.seen == []


def test_a_server_that_never_delivers_is_an_error_not_a_hang(tmp_path: Path, server: str) -> None:
    Flaky.PAYLOAD, saved = Flaky.PAYLOAD[:100], Flaky.PAYLOAD
    try:
        original = Flaky.do_GET
        Flaky.do_GET = lambda self: self.connection.close()  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="did not download completely"):
            download.download_resumable(
                server, tmp_path / "x.bin", attempts=2, pause=lambda s: None
            )
    finally:
        Flaky.do_GET = original  # type: ignore[method-assign]
        Flaky.PAYLOAD = saved


def test_a_lost_network_is_waited_out_with_growing_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DNS failures during a real download came in a burst of seconds: retrying at once used up the
    retries before the network was back."""
    calls = {"n": 0}
    real = urllib.request.urlopen

    def flaky_open(request: object, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 3:
            raise urllib.error.URLError("nodename nor servname provided, or not known")
        return real(request, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(urllib.request, "urlopen", flaky_open)
    pauses: list[float] = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    Flaky.cut_first, Flaky.seen = False, []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/f.bin"
        download.download_resumable(url, tmp_path / "f.bin", pause=pauses.append)
    finally:
        httpd.shutdown()
    assert (tmp_path / "f.bin").read_bytes() == Flaky.PAYLOAD
    assert pauses == [10.0, 20.0, 40.0]  # three misses in a row, each waited out longer


def test_a_slow_link_that_keeps_delivering_is_never_abandoned(tmp_path: Path, server: str) -> None:
    """Many drops are fine as long as each makes progress (the old counter gave up after 5)."""
    chunks = {"n": 0}
    original = Flaky.do_GET

    def cut_often(self: Flaky) -> None:
        start = int(self.headers.get("Range", "bytes=0-").removeprefix("bytes=").rstrip("-"))
        chunks["n"] += 1
        body = Flaky.PAYLOAD[start : start + 100_000]  # 100 kB per connection, then it drops
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(Flaky.PAYLOAD) - start))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.connection.close()

    Flaky.do_GET = cut_often  # type: ignore[method-assign]
    try:
        download.download_resumable(server, tmp_path / "s.bin", attempts=3, pause=lambda s: None)
    finally:
        Flaky.do_GET = original  # type: ignore[method-assign]
    assert (tmp_path / "s.bin").read_bytes() == Flaky.PAYLOAD and chunks["n"] > 3
