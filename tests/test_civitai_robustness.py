"""
Civitai lookups must never prevent an image from being saved: network errors and
unexpected .civitai.info content fall back to plain hashes, and failed lookups are
not repeated for every image.
"""
import pytest
import requests

from image_saver_under_test import utils_civitai
from image_saver_under_test.utils_civitai import get_civitai_metadata


@pytest.fixture(autouse=True)
def no_failed_lookups():
    utils_civitai._failed_lookups.clear()
    yield
    utils_civitai._failed_lookups.clear()


def metadata_for_model(tmp_path):
    return get_civitai_metadata("model", str(tmp_path / "model.safetensors"), "abcdef1234", {}, {}, {}, True)


def test_network_error_falls_back_to_model_hash(tmp_path, monkeypatch):
    def broken_download(*args, **kwargs):
        raise requests.exceptions.ChunkedEncodingError("connection broken")
    monkeypatch.setattr(requests, "get", broken_download)

    resources, hashes, add_model_hash = metadata_for_model(tmp_path)
    assert resources == []
    assert add_model_hash == "ABCDEF1234"


@pytest.mark.parametrize("content", ["{}", '{"model": "not a dict", "id": 1}', "[]", "not json"])
def test_unexpected_civitai_info_file_falls_back_to_model_hash(tmp_path, content):
    (tmp_path / "model.civitai.info").write_text(content)
    resources, _, add_model_hash = metadata_for_model(tmp_path)
    assert resources == []
    assert add_model_hash == "ABCDEF1234"


def test_unexpected_exception_falls_back_to_hashes(monkeypatch):
    def failing_info(path, hash):
        raise RuntimeError("boom")
    monkeypatch.setattr(utils_civitai, "get_civitai_info", failing_info)

    resources, hashes, add_model_hash = get_civitai_metadata(
        "model", "model.safetensors", "abcdef1234", {"LORA:style": ("style.safetensors", 0.8, "fedcba4321")}, {}, {}, True)
    assert resources == []
    assert add_model_hash == "ABCDEF1234"
    assert hashes == {"LORA:style": "FEDCBA4321"}


def test_failed_lookup_is_not_repeated(tmp_path, monkeypatch):
    calls = []
    def not_found(url, **kwargs):
        calls.append(url)
        response = requests.models.Response()
        response.status_code = 404
        return response
    monkeypatch.setattr(requests, "get", not_found)

    metadata_for_model(tmp_path)
    metadata_for_model(tmp_path)
    assert len(calls) == 1


def test_failed_lookup_is_retried_after_a_while(tmp_path, monkeypatch):
    calls = []
    def not_found(url, **kwargs):
        calls.append(url)
        response = requests.models.Response()
        response.status_code = 404
        return response
    monkeypatch.setattr(requests, "get", not_found)

    metadata_for_model(tmp_path)
    utils_civitai._failed_lookups["ABCDEF1234"] -= utils_civitai.RETRY_FAILED_LOOKUP_AFTER + 1
    metadata_for_model(tmp_path)
    assert len(calls) == 2
