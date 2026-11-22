"""
The "with Name" loaders store the model name on the MODEL, and Get Model Name reads it back.
"""
import comfy.sd
import folder_paths
import pytest

from image_saver_under_test.nodes_loaders import CheckpointLoaderWithName, GetModelName, UNETLoaderWithName


class FakeModelPatcher:
    """Mirrors ModelPatcher's attachments: get/set by key, copied by clone()."""

    def __init__(self):
        self.attachments = {}

    def set_attachments(self, key, attachment):
        self.attachments[key] = attachment

    def get_attachment(self, key):
        return self.attachments.get(key, None)

    def clone(self):
        n = FakeModelPatcher()
        n.attachments = dict(self.attachments)
        return n


@pytest.fixture
def loaders(monkeypatch):
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: f"/models/{folder}/{name}", raising=False)
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: f"/models/{folder}/{name}", raising=False)
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda folder: [f"/models/{folder}"], raising=False)
    monkeypatch.setattr(comfy.sd, "load_checkpoint_guess_config", lambda path, **kw: (FakeModelPatcher(), "CLIP", "VAE", None), raising=False)
    monkeypatch.setattr(comfy.sd, "load_diffusion_model", lambda path, **kw: FakeModelPatcher(), raising=False)


def get_name(model):
    return GetModelName().get_model_name(model)[0]


def test_checkpoint_loader_name_reaches_getter(loaders):
    model, clip, vae, name = CheckpointLoaderWithName().load_checkpoint("sdxl/pony.safetensors")
    assert name == "sdxl/pony.safetensors"
    assert get_name(model) == "sdxl/pony.safetensors"


def test_unet_loader_name_reaches_getter(loaders):
    model, name = UNETLoaderWithName().load_unet("flux/flux1-dev.safetensors", "default")
    assert name == "flux/flux1-dev.safetensors"
    assert get_name(model) == "flux/flux1-dev.safetensors"


def test_name_survives_patching(loaders):
    model, *_ = CheckpointLoaderWithName().load_checkpoint("pony.safetensors")
    patched = model.clone().clone()  # e.g. two LoRA loaders
    assert get_name(patched) == "pony.safetensors"


def test_model_from_other_loader_gives_empty_name():
    assert get_name(FakeModelPatcher()) == ""
