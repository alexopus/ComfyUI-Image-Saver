"""
Cache files for manually added hashes are named after a filename from Civitai's API
(or from manual-hashes.json). That name must not be able to point outside the cache folder.
"""
import json

import pytest

import folder_paths
from image_saver_under_test import utils_civitai


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setattr(folder_paths, "models_dir", str(models), raising=False)
    return models


def civitai_content(file_name):
    return {"modelId": 1, "id": 2, "name": "v1", "model": {"name": "Some LoRA", "type": "LORA"},
            "files": [{"name": file_name, "hashes": {"AutoV2": "ABCDEF1234"}}]}


@pytest.mark.parametrize("file_name", ["../../escaped.safetensors", "/tmp/escaped.safetensors", "..\\..\\escaped.safetensors"])
def test_downloaded_filename_stays_in_cache_folder(file_name, tmp_path, models_dir, monkeypatch):
    monkeypatch.setattr(utils_civitai, "download_model_info", lambda path, hash: civitai_content(file_name))

    assert utils_civitai.get_civitai_info(None, "ABCDEF1234")["id"] == 2

    written = list(tmp_path.rglob("*.civitai.info"))
    assert written == [models_dir / "image-saver" / "escaped.civitai.info"]
    manual_list = json.loads((models_dir / "image-saver" / "manual-hashes.json").read_text())
    assert manual_list["ABCDEF1234"]["filename"] == "escaped.safetensors"


@pytest.mark.parametrize("file_name", ["..", "", None])
def test_unusable_downloaded_filename_is_not_cached(file_name, tmp_path, monkeypatch):
    monkeypatch.setattr(utils_civitai, "download_model_info", lambda path, hash: civitai_content(file_name))

    assert utils_civitai.get_civitai_info(None, "ABCDEF1234")["id"] == 2
    assert list(tmp_path.rglob("*.civitai.info")) == []


def test_manual_list_filename_cannot_escape(tmp_path, models_dir):
    outside = tmp_path / "secret.civitai.info"
    outside.write_text(json.dumps({"id": 666}))
    cache = models_dir / "image-saver"
    cache.mkdir(parents=True)
    (cache / "manual-hashes.json").write_text(json.dumps({"ABCDEF1234": {"filename": "../../secret.safetensors"}}))
    (cache / "secret.civitai.info").write_text(json.dumps({"id": 2}))

    assert utils_civitai.get_civitai_info(None, "ABCDEF1234") == {"id": 2}
