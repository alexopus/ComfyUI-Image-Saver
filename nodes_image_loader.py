"""
"Load Image with Metadata": ComfyUI's Load Image plus the generation metadata stored in the file,
as METADATA for the Image Saver nodes (re-saving keeps the original A1111 parameters unchanged).

The node delegates image loading, change detection and validation to ComfyUI's LoadImage instead of
subclassing it, so it doesn't inherit LoadImage's Essentials category and search aliases.
A small HTTP route lets the frontend show the metadata as soon as an image is selected.
"""
import asyncio
import os
from typing import Any

import folder_paths
from nodes import LoadImage

from .nodes import Metadata
from .saver.image_metadata import ImageMetadata, display_rows, read_image_metadata
from .utils_civitai import CIVITAI_SAMPLER_MAP

try:
    from comfy.samplers import KSampler
    SCHEDULERS: list[str] = list(KSampler.SCHEDULERS)
except Exception:  # outside ComfyUI (tests)
    SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]

# "DPM++ 2M" -> "dpmpp_2m" (for duplicates like UniPC the last entry wins, which is fine)
COMFY_SAMPLER_NAMES = {display: name for name, display in CIVITAI_SAMPLER_MAP.items()}


def split_sampler(sampler: str, scheduler: str = "") -> tuple[str, str]:
    """Undo get_civitai_sampler_name: "DPM++ 2M Karras" -> ("dpmpp_2m", "karras"), "foo_beta" -> ("foo", "beta").
    `scheduler` is a separate "Schedule type"/"Scheduler" value, if the source has one."""
    scheduler = scheduler.strip().lower()
    scheduler = scheduler if scheduler in SCHEDULERS else ""
    sampler = sampler.strip()
    for suffix in ("Karras", "Exponential"):
        if sampler.endswith(f" {suffix}") and sampler[:-len(suffix) - 1] in COMFY_SAMPLER_NAMES:
            return COMFY_SAMPLER_NAMES[sampler[:-len(suffix) - 1]], scheduler or suffix.lower()
    if sampler in COMFY_SAMPLER_NAMES:
        return COMFY_SAMPLER_NAMES[sampler], scheduler or "normal"
    if not scheduler:
        for candidate in sorted(SCHEDULERS, key=len, reverse=True):
            if sampler.endswith(f"_{candidate}"):
                return sampler[:-len(candidate) - 1], candidate
    return sampler, scheduler or "normal"


def _number(value: Any, kind: type, default):
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return default


def saver_hashes(modelname: str, settings: dict[str, Any]) -> str:
    """The "Model hash" and "Hashes" entries as a comma-separated list for additional_hashes ("name:HASH").
    Resources only listed under "Civitai resources" have no hash and are not included."""
    entries: list[str] = []
    seen: set[str] = set()

    def add(name: str, value: Any):
        if isinstance(value, str) and value and value.lower() not in seen:
            seen.add(value.lower())
            entries.append(f"{name}:{value}" if name else value)

    add(modelname, settings.get("Model hash"))
    hashes = settings.get("Hashes")
    if isinstance(hashes, dict):
        for key, value in hashes.items():
            name = modelname if key == "model" else str(key).split(":", 1)[-1]
            add(name, value)
    return ",".join(entries)


def to_saver_metadata(metadata: ImageMetadata) -> Metadata | None:
    if metadata.source == "none":
        return None
    settings = metadata.settings
    sampler_name, scheduler_name = split_sampler(str(settings.get("Sampler", "")),
                                                 str(settings.get("Schedule type", settings.get("Scheduler", ""))))
    size = str(settings.get("Size", ""))
    width, _, height = size.partition("x")
    modelname = str(settings.get("Model", ""))
    return Metadata(
        modelname=modelname,
        positive=metadata.positive,
        negative=metadata.negative,
        width=_number(width, int, metadata.width),
        height=_number(height, int, metadata.height),
        seed=_number(settings.get("Seed"), int, 0),
        steps=_number(settings.get("Steps"), int, 20),
        cfg=_number(settings.get("CFG scale"), float, 7.0),
        sampler_name=sampler_name,
        scheduler_name=scheduler_name,
        denoise=_number(settings.get("Denoising strength"), float, 1.0),
        clip_skip=_number(settings.get("Clip skip"), int, 0),
        custom="",
        additional_hashes="",
        ckpt_path="",
        # Written unchanged by the savers, so a re-saved image keeps the original parameters and resources
        a111_params=metadata.parameters if metadata.source == "parameters" else "",
        final_hashes=saver_hashes(modelname, settings),
    )


def metadata_warning(filename: str, metadata: ImageMetadata) -> str:
    if metadata.source == "parameters":
        return ""
    if metadata.source == "workflow":
        return ("No A1111 parameters in this file: values were read from the embedded ComfyUI workflow (best effort). "
                "A saver connected to the metadata output writes no parameters.")
    warning = "No generation metadata found in this file."
    if os.path.basename(filename.split(" [")[0]).startswith("clipspace-"):
        warning += " It is a copy made by the mask editor, which doesn't keep the original's metadata."
    return warning


class LoadImageWithMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return LoadImage.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE", "MASK", "METADATA", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "metadata", "positive", "negative")
    OUTPUT_TOOLTIPS = (
        "image (same as Load Image)",
        "mask from the alpha channel (same as Load Image)",
        "generation metadata of the file for Image Saver Simple: re-saving keeps the original parameters and resources. Empty if the file has none.",
        "positive prompt",
        "negative prompt",
    )
    FUNCTION = "load_image"
    CATEGORY = "ImageSaver"
    DESCRIPTION = ("Load Image plus the generation metadata stored in the file (PNG, JPEG, WebP, AVIF from Image Saver, "
                   "A1111/Forge or ComfyUI). Connect metadata to Image Saver Simple to keep it when saving a processed version.")
    SEARCH_ALIASES = ["load image metadata", "read image metadata", "image prompt reader", "load generation parameters"]

    def load_image(self, image: str):
        pixels, mask = LoadImage().load_image(image)
        try:
            metadata = read_image_metadata(folder_paths.get_annotated_filepath(image))
        except Exception as e:
            print(f"ComfyUI-Image-Saver: Could not read metadata from '{image}': {type(e).__name__}: {e}")
            metadata = ImageMetadata()
        warning = metadata_warning(image, metadata)
        if warning:
            print(f"ComfyUI-Image-Saver: {image}: {warning}")
        return (pixels, mask, to_saver_metadata(metadata), metadata.positive, metadata.negative)

    @classmethod
    def IS_CHANGED(cls, image: str):
        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        return LoadImage.VALIDATE_INPUTS(image)


def _allowed_roots() -> list[str]:
    roots = []
    for getter in ("get_input_directory", "get_output_directory", "get_temp_directory"):
        get_dir = getattr(folder_paths, getter, None)
        if get_dir is not None:
            roots.append(os.path.realpath(get_dir()))
    return roots


def metadata_response(filename: str) -> tuple[int, dict[str, Any]]:
    """Response for the frontend's live preview: (HTTP status, JSON body)."""
    if not filename:
        return 400, {"error": "missing filename"}
    try:
        path = os.path.realpath(folder_paths.get_annotated_filepath(filename))
    except ValueError:
        return 403, {"error": "invalid path"}
    if not any(os.path.commonpath([root, path]) == root for root in _allowed_roots() if os.path.splitdrive(root)[0] == os.path.splitdrive(path)[0]):
        return 403, {"error": "invalid path"}
    if not os.path.isfile(path):
        return 404, {"error": "file not found"}
    try:
        metadata = read_image_metadata(path)
    except Exception as e:
        return 200, {"source": "none", "parameters": "", "rows": [], "warning": f"Could not read the file: {type(e).__name__}: {e}"}
    return 200, {
        "source": metadata.source,
        "parameters": metadata.parameters,
        "rows": display_rows(metadata) if metadata.source != "none" else [],
        "warning": metadata_warning(filename, metadata),
    }


try:
    from aiohttp import web
    from server import PromptServer
except ImportError:  # outside ComfyUI (tests)
    PromptServer = None

if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/image_saver/image_metadata")
    async def image_metadata_route(request):
        status, body = await asyncio.to_thread(metadata_response, request.rel_url.query.get("filename", ""))
        return web.json_response(body, status=status)
