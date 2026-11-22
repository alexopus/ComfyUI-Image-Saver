"""
Test harness for exercising ComfyUI-dependent modules (nodes.py, utils.py)
without a full ComfyUI installation.

ComfyUI normally supplies `folder_paths`, `nodes`, and the `comfy.*` package
tree to custom node plugins at load time. Outside of ComfyUI we stand in
minimal stubs for those here.

We load nodes.py/utils.py directly under a synthetic parent package instead
of importing this repository's own __init__.py, because __init__.py eagerly
pulls in nodes (checkpoint/UNET loaders, sampler selectors, etc.) that need
much more of ComfyUI's internals than the plain file-path logic under test.
"""
import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_NAME = "image_saver_under_test"


def _stub_module(name: str, **attrs) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubLoadImage:
    """Stands in for ComfyUI's LoadImage: returns markers instead of tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (["stub.png"], {"image_upload": True})}}

    def load_image(self, image):
        return ("IMAGE:" + image, "MASK:" + image)

    @classmethod
    def IS_CHANGED(cls, image):
        return "changed:" + image

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True


def _install_comfyui_stubs() -> None:
    _stub_module("torch", Tensor=object)
    _stub_module("tqdm", tqdm=lambda iterable=None, **kw: iterable if iterable is not None else [])
    _stub_module(
        "folder_paths",
        output_directory=str(REPO_ROOT),  # each test overrides this via monkeypatch
        supported_pt_extensions={".safetensors", ".ckpt", ".pt"},
        get_full_path=lambda *a, **kw: None,
    )
    _stub_module("nodes", MAX_RESOLUTION=16384, LoadImage=_StubLoadImage)
    comfy = _stub_module("comfy")
    comfy.__path__ = []  # mark as a package so `comfy.sd1_clip` resolves under it
    comfy.samplers = _stub_module("comfy.samplers", KSampler=type("KSampler", (), {
        "SAMPLERS": ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"],
        "SCHEDULERS": ["normal", "karras", "exponential", "simple", "beta"],
    }))
    comfy.sd = _stub_module("comfy.sd")  # loader functions are monkeypatched per test
    _stub_module(
        "comfy.sd1_clip",
        escape_important=lambda *a, **kw: None,
        unescape_important=lambda *a, **kw: None,
        token_weights=lambda *a, **kw: None,
    )


def _register_plugin_package() -> None:
    if PKG_NAME in sys.modules:
        return
    package = types.ModuleType(PKG_NAME)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[PKG_NAME] = package


_install_comfyui_stubs()
_register_plugin_package()
