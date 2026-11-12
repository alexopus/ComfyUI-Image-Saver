"""
CivitaiHashFetcher must return a tuple on every path: ComfyUI indexes the return value by
output position, so a bare string would become its first character.
"""
import requests

from image_saver_under_test.civitai_nodes import CivitaiHashFetcher


class FakeResponse:
    def __init__(self, data):
        self.status_code = 200
        self._data = data

    def json(self):
        return self._data


def fake_get(url, params=None, timeout=None):
    if url.endswith("/models"):
        return FakeResponse({"items": [{"name": "My LoRA", "modelVersions": [{"id": 7, "name": "v1"}]}]})
    return FakeResponse({"files": [{"hashes": {"AutoV3": "0123456789AB"}}]})


def test_cached_result_is_a_tuple(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_get)
    node = CivitaiHashFetcher()

    assert node.get_autov3_hash("someone", "My LoRA") == ("0123456789AB",)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should use cache")))
    assert node.get_autov3_hash("someone", "My LoRA") == ("0123456789AB",)
