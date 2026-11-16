"""
"Edit Metadata": change or remove parts of finished METADATA (e.g. from Load Image with Metadata or
Image Saver Metadata) without rebuilding it. Only the changed parts of the A1111 parameters are rewritten;
everything else, including hashes and Civitai resources, is kept as written. No hashing, no Civitai lookups.

Inputs follow Edit Image Saver Pipe: text fields use the [original] placeholder, numbers and
sampler/scheduler only apply when connected.
"""
import dataclasses
import re
from typing import Any

from .nodes import Metadata, parse_checkpoint_name_without_extension
from .nodes_pipe import EditImageSaverPipe, apply_string_edit
from .saver.image_metadata import edit_parameters
from .utils_civitai import get_civitai_sampler_name

EDIT_FIELDS = ("modelname", "positive", "negative", "width", "height", "seed", "steps", "cfg",
               "sampler_name", "scheduler_name", "denoise", "clip_skip")


class EditMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        pipe_fields = EditImageSaverPipe.INPUT_TYPES()["optional"]
        optional: dict[str, Any] = {key: pipe_fields[key] for key in EDIT_FIELDS if key in pipe_fields}
        optional["modelname"] = (optional["modelname"][0], {**optional["modelname"][1],
            "tooltip": "model name shown in the parameters and used by %model in filenames. "
                       "Doesn't change hashes or Civitai resources (remove them with remove_settings if needed). "
                       "Supports the [original] placeholder; empty removes it."})
        for key in ("positive", "negative"):
            optional[key] = (optional[key][0], {**optional[key][1],
                "tooltip": f"{key} prompt. '[original]' keeps it, '[original], more' appends, other text replaces it, empty removes it."})
        optional["remove_settings"] = ("STRING", {"default": "", "multiline": False,
            "tooltip": "comma-separated A1111 settings to remove, e.g. 'Civitai resources, Hashes, Model hash' or 'Version, Lora hashes'"})
        optional["remove_resources"] = ("STRING", {"default": "", "multiline": False,
            "tooltip": "comma-separated resources to remove from Hashes, Civitai resources, Lora/TI hashes and the hashes output: "
                       "names (e.g. 'EasyNegative' or the Civitai name), hashes, Civitai version IDs or AIRs"})
        return {"required": {"metadata": ("METADATA", {"tooltip": "metadata to edit"})}, "optional": optional}

    RETURN_TYPES = ("METADATA",)
    RETURN_NAMES = ("metadata",)
    OUTPUT_TOOLTIPS = ("the edited metadata, for Image Saver Simple or another Edit Metadata",)
    FUNCTION = "edit"
    CATEGORY = "ImageSaver"
    DESCRIPTION = ("Change or remove parts of existing metadata, e.g. from Load Image with Metadata. Only what you change "
                   "is rewritten; hashes, Civitai resources and other settings stay as they are. Numbers and "
                   "sampler/scheduler only change when connected.")

    def edit(self, metadata: Metadata | None, remove_settings: str = "", remove_resources: str = "",
             **kwargs: Any) -> tuple[Metadata | None]:
        if metadata is None:
            print("ComfyUI-Image-Saver: Edit Metadata received no metadata, passing it through.")
            return (None,)

        fields: dict[str, Any] = {}
        changes: dict[str, str | None] = {}

        positive = apply_string_edit(kwargs.get("positive"), metadata.positive)
        negative = apply_string_edit(kwargs.get("negative"), metadata.negative)
        if positive != metadata.positive:
            fields["positive"] = positive
        if negative != metadata.negative:
            fields["negative"] = negative

        modelname = apply_string_edit(kwargs.get("modelname"), metadata.modelname)
        if modelname != metadata.modelname:
            fields["modelname"] = modelname
            changes["Model"] = parse_checkpoint_name_without_extension(modelname) if modelname else None

        def connected(key: str) -> Any:
            return kwargs.get(key)

        if (seed := connected("seed")) is not None:
            fields["seed"] = int(seed)
            changes["Seed"] = str(int(seed))
        if (steps := connected("steps")) is not None:
            fields["steps"] = int(steps)
            changes["Steps"] = str(int(steps))
        if (cfg := connected("cfg")) is not None:
            fields["cfg"] = float(cfg)
            changes["CFG scale"] = str(float(cfg))
        if (denoise := connected("denoise")) is not None:
            fields["denoise"] = float(denoise)
            # Like A1111: only written when it isn't a full denoise
            changes["Denoising strength"] = str(float(denoise)) if float(denoise) != 1.0 else None
        if (clip_skip := connected("clip_skip")) is not None:
            fields["clip_skip"] = int(clip_skip)
            changes["Clip skip"] = str(abs(int(clip_skip))) if int(clip_skip) != 0 else None
        if connected("width") is not None or connected("height") is not None:
            width = int(connected("width") if connected("width") is not None else metadata.width)
            height = int(connected("height") if connected("height") is not None else metadata.height)
            fields["width"], fields["height"] = width, height
            changes["Size"] = f"{width}x{height}"
        if connected("sampler_name") is not None or connected("scheduler_name") is not None:
            sampler = str(connected("sampler_name") if connected("sampler_name") is not None else metadata.sampler_name)
            scheduler = str(connected("scheduler_name") if connected("scheduler_name") is not None else metadata.scheduler_name)
            fields["sampler_name"], fields["scheduler_name"] = sampler, scheduler
            changes["Sampler"] = get_civitai_sampler_name(sampler.replace("_gpu", ""), scheduler)
            changes["Schedule type"] = None  # newer A1111 field; the scheduler is part of Sampler now

        remove = [key.strip() for key in remove_settings.split(",") if key.strip()]
        resources = [name.strip() for name in remove_resources.split(",") if name.strip()]
        final_hashes = drop_named_hashes(metadata.final_hashes, resources)
        if metadata.a111_params:
            fields["a111_params"] = edit_parameters(metadata.a111_params, fields.get("positive"), fields.get("negative"),
                                                    changes, remove, resources)
            final_hashes = drop_removed_hashes(final_hashes, metadata.a111_params, fields["a111_params"])
        if final_hashes != metadata.final_hashes:
            fields["final_hashes"] = final_hashes

        return (dataclasses.replace(metadata, **fields),)


HASH_TOKEN = re.compile(r"\b[0-9a-fA-F]{8,16}\b")


def drop_named_hashes(final_hashes: str, names: list[str]) -> str:
    """Drop hashes output entries ("name:HASH[:weight]") whose name or hash is in `names`."""
    wanted = {name.casefold() for name in names}
    if not wanted:
        return final_hashes
    return ",".join(entry for entry in final_hashes.split(",")
                    if entry.strip() and not any(part.strip().casefold() in wanted for part in entry.split(":")))


def drop_removed_hashes(final_hashes: str, old_parameters: str, new_parameters: str) -> str:
    """Drop the hashes output entries ("name:HASH[:weight]") whose hash was in the parameters before the edit but
    isn't anymore, e.g. after removing "Hashes" (the model stays if "Model hash" is kept). Entries that never
    appeared in the parameters (e.g. resources resolved via Civitai) are kept."""
    old = {token.lower() for token in HASH_TOKEN.findall(old_parameters)}
    new = {token.lower() for token in HASH_TOKEN.findall(new_parameters)}
    gone = old - new
    if not gone:
        return final_hashes
    kept = [entry for entry in final_hashes.split(",")
            if entry.strip() and not any(part.strip().lower() in gone for part in entry.split(":"))]
    return ",".join(kept)
