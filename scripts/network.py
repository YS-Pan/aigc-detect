"""Download public artifacts, direct first, with environment-proxy fallback.

Never changes process or system proxy settings, disables TLS checks, or sends text
for inference. A failed direct host is remembered during this download process.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FALLBACK = urllib.request.build_opener()
FAILED_DIRECT = {}
EVENTS = []


def open_public(url, *, timeout=15, headers=None):
    host = urllib.parse.urlparse(url).netloc
    request = urllib.request.Request(url, headers={"User-Agent": "local-aigc-setup/0.1", **(headers or {})})
    if host not in FAILED_DIRECT:
        try:
            response = DIRECT.open(request, timeout=timeout)
            EVENTS.append({"host": host, "route": "direct", "result": "ok"})
            return response
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 404):
                raise
            FAILED_DIRECT[host] = type(exc).__name__ + f": HTTP {exc.code}"
        except (OSError, urllib.error.URLError) as exc:
            FAILED_DIRECT[host] = type(exc).__name__
        print(f"Direct connection to {host} failed ({FAILED_DIRECT[host]}); trying configured proxy.", flush=True)
        EVENTS.append({"host": host, "route": "direct", "result": "failed", "reason": FAILED_DIRECT[host]})
    response = FALLBACK.open(request, timeout=60)
    EVENTS.append({"host": host, "route": "environment_proxy_fallback", "result": "ok"})
    return response


def get_json(url):
    with open_public(url) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_sha=None, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        digest = sha256(destination)
        if expected_sha is None or digest == expected_sha:
            return {"sha256": digest, "bytes": destination.stat().st_size, "cached": True}
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        try:
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
