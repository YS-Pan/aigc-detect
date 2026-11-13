"""Download public artifacts, direct first, with environment-proxy fallback.

Never changes process or system proxy settings, disables TLS checks, or sends text
for inference. A failed direct host is remembered during this download process.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Resolve each redirect separately. A blocked metadata origin must not force a
# reachable multi-GB CDN download through the proxy as well.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
FALLBACK = urllib.request.build_opener(_NoRedirect())
FAILED_DIRECT = {}
EVENTS = []


def _open_host(url, *, timeout=15, headers=None):
    host = urllib.parse.urlparse(url).netloc
    request = urllib.request.Request(url, headers={"User-Agent": "local-aigc-setup/0.1", **(headers or {})})
    if host not in FAILED_DIRECT:
        try:
            response = DIRECT.open(request, timeout=timeout)
            EVENTS.append({"host": host, "route": "direct", "result": "ok"})
            return response
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                EVENTS.append({"host": host, "route": "direct", "result": "ok", "status": exc.code})
                return exc
            if exc.code in (400, 401, 404):
                raise
            FAILED_DIRECT[host] = type(exc).__name__ + f": HTTP {exc.code}"
        except (OSError, urllib.error.URLError) as exc:
            FAILED_DIRECT[host] = type(exc).__name__
        print(f"Direct connection to {host} failed ({FAILED_DIRECT[host]}); trying configured proxy.", flush=True)
        EVENTS.append({"host": host, "route": "direct", "result": "failed", "reason": FAILED_DIRECT[host]})
    try:
        response = FALLBACK.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        response = exc
    EVENTS.append({"host": host, "route": "environment_proxy_fallback", "result": "ok"})
    return response


def open_public(url, *, timeout=15, headers=None):
    for _ in range(10):
        response = _open_host(url, timeout=timeout, headers=headers)
        if response.status not in (301, 302, 303, 307, 308):
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise RuntimeError("Public resource redirect has no Location")
        url = urllib.parse.urljoin(url, location)
        if urllib.parse.urlparse(url).scheme != "https":
            raise RuntimeError("Public resource redirected outside HTTPS")
    raise RuntimeError("Too many public resource redirects")


def get_json(url):
    with open_public(url) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parallel_download(url, partial, expected_sha, expected_size, workers, *, part_size=64 * 1024 * 1024):
    """Resume independently checked HTTP byte ranges, then assemble in order."""
    directory = partial.with_name(partial.name + ".parts")
    directory.mkdir(exist_ok=True)
    metadata = {"url": url, "sha256": expected_sha, "bytes": expected_size, "part_bytes": part_size}
    identity = directory / "identity.json"
    if identity.exists() and json.loads(identity.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("Partial byte ranges belong to a different resource")
    identity.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    ranges = [(start, min(start + part_size, expected_size)) for start in range(0, expected_size, part_size)]
    paths = [directory / f"{index:06}.part" for index in range(len(ranges))]
    # Preserve progress from the previous sequential downloader.
    if partial.exists():
        prefix = partial.stat().st_size
        with partial.open("rb") as old:
            for (start, end), path in zip(ranges, paths):
                count = min(end, prefix) - start
                if count <= 0:
                    break
                if not path.exists() or path.stat().st_size < count:
                    old.seek(start)
                    path.write_bytes(old.read(count))

    def fetch(index):
        start, end = ranges[index]
        path = paths[index]
        for attempt in range(3):
            offset = path.stat().st_size if path.exists() else 0
            if offset == end - start:
                return
            if offset > end - start:
                raise RuntimeError("Oversized partial byte range")
            try:
                with open_public(url, headers={"Range": f"bytes={start + offset}-{end - 1}"}) as response:
                    wanted = f"bytes {start + offset}-{end - 1}/{expected_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise RuntimeError("Server did not return the requested exact byte range")
                    with path.open("ab") as handle:
                        remaining = end - start - offset
                        while remaining:
                            block = response.read(min(2 * 1024 * 1024, remaining))
                            if not block:
                                raise RuntimeError("Truncated byte range")
                            handle.write(block)
                            remaining -= len(block)
                return
            except Exception:
                if attempt == 2:
                    raise

    fetch(0)  # Establish range support before dispatching other requests.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, index) for index in range(1, len(ranges))}
        while pending:
            done, pending = wait(pending, timeout=15)
            for future in done:
                future.result()
            count = sum(path.stat().st_size for path in paths if path.exists())
            print(f"{partial.parent.name}/{partial.name}: {count / 1e6:.1f}/{expected_size / 1e6:.1f} MB ({workers} direct-first ranges)", flush=True)
    assembled = partial.with_name(partial.name + ".assembled")
    with assembled.open("wb") as output:
        for (start, end), path in zip(ranges, paths):
            if path.stat().st_size != end-start:
                raise RuntimeError("Incomplete byte range during assembly")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
    os.replace(assembled, partial)
    return paths, identity, directory


def download(url, destination, expected_sha=None, expected_size=None, *, workers=1):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        digest = sha256(destination)
        if expected_sha is None or digest == expected_sha:
            return {"sha256": digest, "bytes": destination.stat().st_size, "cached": True}
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        try:
            parts = None
            if workers > 1 and expected_sha and expected_size and expected_size >= 256 * 1024 * 1024:
                parts = _parallel_download(url, partial, expected_sha, expected_size, workers)
                digest = sha256(partial)
                if digest != expected_sha:
                    partial.unlink()
                    for path in parts[0]:
                        path.unlink(missing_ok=True)
                    raise RuntimeError("Parallel download SHA-256 mismatch")
                os.replace(partial, destination)
                for path in parts[0]:
                    path.unlink()
                parts[1].unlink()
                parts[2].rmdir()
                print(f"Saved {destination.parent.name}/{destination.name} ({expected_size / 1e6:.1f} MB)", flush=True)
                return {"sha256": digest, "bytes": destination.stat().st_size, "cached": False, "range_workers": workers}
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with open_public(url, headers=headers) as response:
                resumed = offset and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned unexpected resume range")
                count = offset if resumed else 0
                last_report = time.monotonic()
                with partial.open("ab" if resumed else "wb") as handle:
                    while block := response.read(2 * 1024 * 1024):
                        handle.write(block)
                        count += len(block)
                        if time.monotonic() - last_report > 15:
                            print(f"{destination.parent.name}/{destination.name}: {count / 1e6:.1f} MB", flush=True)
                            last_report = time.monotonic()
            digest = sha256(partial)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError("Downloaded size does not match upstream metadata")
            if expected_sha is not None and digest != expected_sha:
                partial.unlink()
                raise RuntimeError("Downloaded SHA-256 does not match upstream metadata")
            os.replace(partial, destination)
            print(f"Saved {destination.parent.name}/{destination.name} ({count / 1e6:.1f} MB)", flush=True)
            return {"sha256": digest, "bytes": destination.stat().st_size, "cached": False}
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/2: {destination.name}: {type(exc).__name__}", flush=True)
    raise RuntimeError("Download failed")
