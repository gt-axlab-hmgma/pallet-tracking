"""A download that survives a flaky connection: resumable, with timeouts and backoff."""

import http.client
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

TIMEOUT = 60  # seconds without a byte before a download is treated as dropped and resumed

Download = Callable[[str, Path], None]


def download_resumable(
    url: str, destination: Path, attempts: int = 8, pause: Callable[[float], None] = time.sleep
) -> None:
    """Download ``url`` to ``destination``, continuing a partial file and retrying on a drop.

    The data goes to ``<destination>.part`` and is renamed only when it is the full length the
    server announced, so a half-finished file is never mistaken for a finished one. A drop (a reset
    connection, a stall of ``TIMEOUT`` seconds, a lost network) waits 10, 20, 40... seconds (at most
    5 minutes) and resumes; it gives up only after ``attempts`` drops in a row that made no
    progress, so a flaky link that keeps delivering a little each time is never abandoned.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    head = urllib.request.Request(url, method="HEAD")
    for tries in range(attempts):
        try:
            with urllib.request.urlopen(head, timeout=TIMEOUT) as response:  # noqa: S310
                total = int(response.headers.get("Content-Length") or 0)
            break
        except (OSError, http.client.HTTPException) as error:
            print(f"  could not reach the server ({error}); try {tries + 1} of {attempts}")
            pause(min(300.0, 10.0 * 2**tries))
    else:
        raise RuntimeError(f"{url} could not be reached")
    if total and destination.exists() and destination.stat().st_size == total:
        return  # an earlier run finished this download (and was cut off while unpacking it)

    failures = 0
    while failures < attempts:
        before = part.stat().st_size if part.exists() else 0
        if total and before == total:
            break
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={before}-"} if before else {}
        )
        have = before
        try:
            with (
                urllib.request.urlopen(request, timeout=TIMEOUT) as response,  # noqa: S310
                part.open("ab" if before else "wb") as out,
            ):
                if before and response.status != 206:  # the server ignored the range: start over
                    out.truncate(0)
                    have = 0
                shown = -1
                while chunk := response.read(1 << 22):
                    out.write(chunk)
                    have += len(chunk)
                    if total and have * 20 // total != shown:
                        shown = have * 20 // total
                        print(f"  {destination.name}: {have / 2**30:.1f} of {total / 2**30:.1f} GB")
        except (OSError, http.client.HTTPException) as error:
            progressed = part.exists() and part.stat().st_size > before
            failures = 0 if progressed else failures + 1
            print(f"  connection dropped ({error}); {failures} failed tries in a row of {attempts}")
            pause(min(300.0, 10.0 * 2**failures))
        else:
            if not total:
                break  # the server did not announce a length: what arrived is all there is
            failures = 0 if part.stat().st_size > before else failures + 1
    if total and (not part.exists() or part.stat().st_size != total):
        raise RuntimeError(f"{url} did not download completely")
    part.replace(destination)
