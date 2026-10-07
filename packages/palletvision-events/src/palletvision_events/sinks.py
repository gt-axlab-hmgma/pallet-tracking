import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

from palletvision_events.record import ZoneEvent, describe


class EventSink(Protocol):
    """Where events go. Writes must be safe to call with an empty batch."""

    def write(self, events: Sequence[ZoneEvent]) -> None: ...

    def close(self) -> None: ...


class StdoutSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def write(self, events: Sequence[ZoneEvent]) -> None:
        for e in events:
            print(
                f"{e.device_ts.isoformat(timespec='milliseconds')}  {describe(e)}",
                file=self._stream,
            )

    def close(self) -> None:
        pass


class JsonlSink:
    """Appends one JSON object per line. Easy to tail, grep and replay."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def write(self, events: Sequence[ZoneEvent]) -> None:
        for e in events:
            self._file.write(json.dumps(e.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
