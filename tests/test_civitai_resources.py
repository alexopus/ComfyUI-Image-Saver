"""
Tests for the "Civitai resources" entries built by utils_civitai.get_civitai_metadata.

Civitai reads `air` (or `modelVersionId`), `type` and `weight`; other tools such as
Stability Matrix need `modelVersionId` and `type` directly (issue #140).
"""
import pytest

from image_saver_under_test import utils_civitai
from image_saver_under_test.utils_civitai import get_civitai_metadata


def civitai_info(version_id, model_name, version_name, model_type, air=None):
    info = {"id": version_id, "name": version_name, "model": {"name": model_name}}
    if model_type is not None:
        info["model"]["type"] = model_type
    if air is not None:
        info["air"] = air
    return info


INFOS = {
    "ckpt_hash": civitai_info(290640, "Pony Diffusion V6 XL", "V6", "Checkpoint", "urn:air:sdxl:checkpoint:civitai:257749@290640"),
    "lora_hash": civitai_info(333590, "Some Style", "Anime 2", "LORA", "urn:air:sdxl:lora:civitai:296485@333590"),
    "locon_hash": civitai_info(111, "Some LoCon", "v1", "LoCon", "urn:air:sdxl:lycoris:civitai:100@111"),
    "embed_hash": civitai_info(5637, "Deep Negative V1.x", "V1 75T", "TextualInversion"),
    "untyped_hash": civitai_info(42, "Old Cache", "v1", None),
}


@pytest.fixture(autouse=True)
def fake_civitai(monkeypatch):
    monkeypatch.setattr(utils_civitai, "get_civitai_info", lambda path, hash: INFOS.get(hash))


def resources_by_version(**kwargs):
    defaults = dict(modelname="pony", ckpt_path="pony.safetensors", modelhash="ckpt_hash",
                    loras={}, embeddings={}, manual_entries={}, download_civitai_data=True)
    resources, _, _ = get_civitai_metadata(**(defaults | kwargs))
    return {r["modelVersionId"]: r for r in resources}


def test_checkpoint_entry_has_version_id_type_and_air():
    resources = resources_by_version()
    assert resources[290640] == {
        "modelName": "Pony Diffusion V6 XL",
        "modelVersionName": "V6",
        "modelVersionId": 290640,
        "type": "checkpoint",
        "air": "urn:air:sdxl:checkpoint:civitai:257749@290640",
    }


def test_lora_types_are_lowercased_civitai_types():
    resources = resources_by_version(loras={
        "style": ("style.safetensors", 0.8, "lora_hash"),
        "locon": ("locon.safetensors", 0.5, "locon_hash"),
    })
    assert resources[333590]["type"] == "lora"
    assert resources[333590]["weight"] == 0.8
    assert resources[111]["type"] == "locon"


def test_embedding_without_air_still_has_version_id():
    resources = resources_by_version(embeddings={"neg": ("neg.pt", 1.0, "embed_hash")})
    assert resources[5637]["type"] == "textualinversion"
    assert "air" not in resources[5637]


def test_missing_model_type_omits_type():
    resources = resources_by_version(manual_entries={"old": (None, None, "untyped_hash")})
    assert "type" not in resources[42]


def test_no_resources_without_civitai_download():
    resources, hashes, add_model_hash = get_civitai_metadata(
        "pony", "pony.safetensors", "ckpt_hash", {}, {}, {}, download_civitai_data=False)
    assert resources == []
    assert add_model_hash == "ckpt_hash"
