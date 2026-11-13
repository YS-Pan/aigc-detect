"""Small HTTP client; no torch import required in the caller's environment."""

import json
import urllib.request


class Client:
    def __init__(self, base_url="http://127.0.0.1:8766", timeout=600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Local text requests should never be routed through the environment proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def health(self):
        return self._request("/health")

    def models(self):
        return self._request("/v1/models")

    def detect(self, text, models=None, **options):
        return self._request("/v1/detect", {"text": text, "models": models, **options})

    def detect_batch(self, texts, models=None, **options):
        return self._request("/v1/detect-batch", {"texts": texts, "models": models, **options})

    def annotate(self, text, models=None, **options):
        return self._request("/v1/annotate", {"text": text, "models": models, **options})
