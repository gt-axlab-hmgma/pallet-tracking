"""A line-by-line log of a training run, one JSON object per line.

Every optimiser step and every epoch is appended as it happens, so you can watch a run with
``tail -f`` and still have the full record if it dies. A resumed run first drops the lines
from the part of the run that was rolled back, so each step appears exactly once.
"""

import json
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, path: Path, keep_up_to_step: int | None = None) -> None:
        """Start a fresh log, or with ``keep_up_to_step`` continue one, keeping steps up to it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        kept: list[str] = []
        if keep_up_to_step is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.strip() and json.loads(line)["step"] <= keep_up_to_step:
                    kept.append(line)
        path.write_text("".join(line + "\n" for line in kept))
        self.path = path

    def write(self, kind: str, **fields: Any) -> None:
        with self.path.open("a") as file:
            file.write(json.dumps({"type": kind, **fields}) + "\n")
