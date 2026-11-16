"""
Read generation metadata back from image files, without ComfyUI.

Sources, in order of preference:
- A1111-style "parameters" text: PNG text chunk (Image Saver, A1111, Forge) or EXIF UserComment (JPEG/WebP/AVIF)
- ComfyUI's embedded "prompt" graph (API format) as a best-effort fallback for images saved by other nodes
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import piexif
import piexif.helper
from PIL import Image

# Tags as written by Image Saver and ComfyUI. EXIF strings carry them as "<tag>:<value>".
KNOWN_TAGS = ("parameters", "prompt", "workflow")


@dataclass
class ImageMetadata:
    tags: dict[str, str] = field(default_factory=dict)  # raw "parameters"/"prompt"/"workflow" strings found in the file
    source: str = "none"  # "parameters", "workflow" or "none"
    positive: str = ""
    negative: str = ""
    settings: dict[str, Any] = field(default_factory=dict)  # A1111 "Key: value" pairs, JSON values decoded
    width: int = 0  # pixel size of the file
    height: int = 0

    @property
    def parameters(self) -> str:
        return self.tags.get("parameters", "")

    @property
    def has_workflow(self) -> bool:
        return bool(self.tags.get("workflow"))


def read_tags(image: Image.Image) -> dict[str, str]:
    tags: dict[str, str] = {}
    for key, value in image.info.items():
        if isinstance(key, str) and key.lower() in KNOWN_TAGS and isinstance(value, str):
            tags.setdefault(key.lower(), value)

    # PNG text chunks after the image data are only read when decoding; check them only if nothing was found
    if not tags and image.format == "PNG":
        for key, value in getattr(image, "text", {}).items():
            if key.lower() in KNOWN_TAGS:
                tags.setdefault(key.lower(), value)

    exif_bytes = image.info.get("exif")
    if exif_bytes:
        try:
            exif = piexif.load(exif_bytes)
        except Exception:
            exif = {}
        comment = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
        if comment and "parameters" not in tags:
            try:
                text = piexif.helper.UserComment.load(comment)
            except Exception:
                text = ""
            if text.strip():
                tags["parameters"] = text
        for value in exif.get("0th", {}).values():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                key, sep, payload = value.partition(":")
                if sep and key.lower() in ("prompt", "workflow"):
                    tags.setdefault(key.lower(), payload)
    return tags


SETTINGS_KEY = re.compile(r"\s*([A-Za-z][\w /().-]*?):\s*")


NEXT_SETTING = re.compile(r",\s*[A-Za-z][\w /().-]*?:\s")


def split_settings(text: str) -> list[tuple[str, str]]:
    """Split an A1111 settings line ("Steps: 20, Sampler: Euler a, Hashes: {...}, ...") into
    (key, raw value text) pairs. JSON and quoted values may contain commas; raw text is kept as is."""
    pairs: list[tuple[str, str]] = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        match = SETTINGS_KEY.match(text, pos)
        if not match:
            break
        key, pos = match.group(1).strip(), match.end()
        if pos < len(text) and text[pos] in '{["':
            try:
                _, end = decoder.raw_decode(text, pos)
                pairs.append((key, text[pos:end]))
                comma = text.find(",", end)
                pos = len(text) if comma == -1 else comma + 1
                continue
            except json.JSONDecodeError:
                pass
        # Plain value: up to the next ", Key:" (so values with commas but no key-like part survive)
        next_key = NEXT_SETTING.search(text, pos)
        end = next_key.start() if next_key else len(text)
        pairs.append((key, text[pos:end].strip()))
        pos = end + 1 if next_key else end
    return pairs


def decode_setting(raw: str) -> Any:
    if raw[:1] in '{["':
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def parse_settings(text: str) -> dict[str, Any]:
    """A1111 settings line as a dict, JSON and quoted values decoded."""
    return {key: decode_setting(raw) for key, raw in split_settings(text)}


def split_parameters(parameters: str) -> tuple[str, str]:
    """Split A1111 parameters into the prompts part and the settings line (the last line starting with "Steps:")."""
    text = parameters.strip("\n")
    settings_start = text.rfind("\nSteps: ")
    if settings_start == -1 and text.startswith("Steps: "):
        settings_start = 0
    if settings_start >= 0:
        return text[:settings_start], text[settings_start:].strip()
    return text, ""


def split_prompts(prompts: str) -> tuple[str, str]:
    parts = re.split(r"(?:^|\n)Negative prompt:\s?", prompts, maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) == 2 else ""


def parse_parameters(parameters: str) -> tuple[str, str, dict[str, Any]]:
    """Split A1111 parameters into positive prompt, negative prompt and settings."""
    prompts, settings_text = split_parameters(parameters)
    positive, negative = split_prompts(prompts)
    return positive, negative, parse_settings(settings_text)


def _dump_like(value: Any, raw: str) -> str:
    """JSON in the style of the original text: compact (Image Saver) or with spaces (A1111)."""
    original = decode_setting(raw)
    for ensure_ascii in (False, True):
        if json.dumps(original, ensure_ascii=ensure_ascii, separators=(", ", ": ")) == raw:
            return json.dumps(value, ensure_ascii=ensure_ascii, separators=(", ", ": "))
        if json.dumps(original, ensure_ascii=ensure_ascii, separators=(",", ":")) == raw:
            return json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _matches(names: set[str], *values: Any) -> bool:
    return any(str(value).strip().casefold() in names for value in values if value not in (None, ""))


def prune_resources(key: str, raw: str, names: set[str]) -> str | None:
    """Remove resources matching `names` (casefolded names, hashes, Civitai version IDs or AIRs) from one setting.
    Returns the new raw value, the unchanged raw value, or None if nothing is left."""
    lowered = key.lower()
    value = decode_setting(raw)
    if lowered == "hashes" and isinstance(value, dict):
        # Keys like "embed:EasyNegative" or "LORA:style" match by the name after the prefix
        kept = {k: v for k, v in value.items() if not _matches(names, k, str(k).split(":", 1)[-1], v)}
        return raw if len(kept) == len(value) else (_dump_like(kept, raw) if kept else None)
    if lowered == "civitai resources" and isinstance(value, list):
        kept = [r for r in value if not (isinstance(r, dict) and _matches(
            names, r.get("modelName"), r.get("modelVersionName"), r.get("modelVersionId"), r.get("air")))]
        return raw if len(kept) == len(value) else (_dump_like(kept, raw) if kept else None)
    if lowered.endswith(" hashes") and isinstance(value, str):
        # A1111 "Lora hashes"/"TI hashes": "name: hash, name2: hash2"
        entries = [entry for entry in value.split(",") if entry.strip()]
        kept = [entry.strip() for entry in entries if not _matches(names, *entry.split(":", 1))]
        return raw if len(kept) == len(entries) else (json.dumps(", ".join(kept), ensure_ascii=False) if kept else None)
    if lowered == "model hash" and _matches(names, raw):
        return None
    return raw


def edit_parameters(parameters: str, positive: str | None = None, negative: str | None = None,
                    settings: dict[str, str | None] | None = None, remove: Iterable[str] = (),
                    remove_resources: Iterable[str] = ()) -> str:
    """Change parts of A1111 parameters in place, keeping everything else as written.

    positive/negative: new prompt, None keeps it (an empty negative drops the "Negative prompt:" line).
    settings: key -> new raw value text, or None to remove the key. New keys go before "Version" or at the end.
    remove: keys to remove (case-insensitive).
    remove_resources: names, hashes, Civitai version IDs or AIRs to remove from "Hashes", "Civitai resources",
    "Lora hashes"/"TI hashes" and "Model hash" (case-insensitive); emptied settings are removed."""
    settings = dict(settings or {})
    removed = {key.strip().lower() for key in remove if key.strip()}
    resources = {name.strip().casefold() for name in remove_resources if name.strip()}
    if positive is None and negative is None and not settings and not removed and not resources:
        return parameters

    prompts, settings_text = split_parameters(parameters)
    if positive is not None or negative is not None:
        old_positive, old_negative = split_prompts(prompts)
        positive = old_positive if positive is None else positive.strip()
        negative = old_negative if negative is None else negative.strip()
        prompts = positive + (f"\nNegative prompt: {negative}" if negative else "")

    pairs: list[tuple[str, str]] = []
    for key, raw in split_settings(settings_text):
        if key.lower() in removed:
            continue
        if key in settings:
            raw = settings.pop(key)
            if raw is None:
                continue
        if resources:
            raw = prune_resources(key, raw, resources)
            if raw is None:
                continue
        pairs.append((key, raw))
    new_pairs = [(key, raw) for key, raw in settings.items() if raw is not None and key.lower() not in removed]
    insert_at = next((i for i, (key, _) in enumerate(pairs) if key == "Version"), len(pairs))
    pairs[insert_at:insert_at] = new_pairs

    settings_line = ", ".join(f"{key}: {raw}" for key, raw in pairs)
    return "\n".join(part for part in (prompts, settings_line) if part)


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))


def _resolve(graph: dict[str, Any], value: Any, names: tuple[str, ...], depth: int = 0) -> Any:
    """Follow links in an API-format graph to a literal, looking for one of `names` in each upstream node."""
    while _is_link(value) and depth < 8:
        node = graph.get(str(value[0]))
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        candidates = [inputs[name] for name in names if name in inputs]
        if not candidates:
            literals = [v for v in inputs.values() if not _is_link(v)]
            candidates = literals if len(literals) == 1 else []
        if not candidates:
            return None
        value, depth = candidates[0], depth + 1
    return None if _is_link(value) else value


TEXT_INPUTS = ("text", "text_g", "string", "value", "prompt")


def parse_comfy_graph(prompt_json: str) -> tuple[str, str, dict[str, Any]]:
    """Best effort: settings of the first full-denoise sampler and the text of its prompts."""
    try:
        graph = json.loads(prompt_json)
    except (json.JSONDecodeError, TypeError):
        return "", "", {}
    if not isinstance(graph, dict):
        return "", "", {}

    def node_id_key(item):
        node_id = item[0]
        return (0, int(node_id)) if str(node_id).isdigit() else (1, str(node_id))

    samplers = [inputs for _, node in sorted(graph.items(), key=node_id_key)
                if isinstance(node, dict) and isinstance(inputs := node.get("inputs"), dict)
                and "steps" in inputs and "cfg" in inputs and "positive" in inputs]
    if not samplers:
        return "", "", {}
    full_denoise = [s for s in samplers if _resolve(graph, s.get("denoise", 1.0), ("denoise",)) in (1, 1.0, None)]
    sampler = (full_denoise or samplers)[0]

    def text(link):
        value = _resolve(graph, link, TEXT_INPUTS)
        return value if isinstance(value, str) else ""

    settings: dict[str, Any] = {}
    for key, names in (("Steps", ("steps",)), ("Sampler", ("sampler_name",)), ("Scheduler", ("scheduler",)),
                       ("CFG scale", ("cfg",)), ("Seed", ("seed", "noise_seed")), ("Denoising strength", ("denoise",))):
        for name in names:
            if name in sampler:
                value = _resolve(graph, sampler[name], (name,) + names)
                if value is not None:
                    settings[key] = value
                break

    for node in graph.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        model = next((inputs[k] for k in ("ckpt_name", "unet_name") if isinstance(inputs.get(k), str)), None)
        if model:
            settings["Model"] = os.path.splitext(os.path.basename(model.replace("\\", "/")))[0]
            break
    return text(sampler.get("positive")), text(sampler.get("negative")), settings


def read_image_metadata(path: str) -> ImageMetadata:
    with Image.open(path) as image:
        result = ImageMetadata(tags=read_tags(image), width=image.width, height=image.height)
    if result.parameters.strip():
        result.source = "parameters"
        result.positive, result.negative, result.settings = parse_parameters(result.parameters)
    elif result.tags.get("prompt"):
        positive, negative, settings = parse_comfy_graph(result.tags["prompt"])
        if positive or negative or settings:
            result.source = "workflow"
            result.positive, result.negative, result.settings = positive, negative, settings
    return result


def _format_value(key: str, value: Any) -> str:
    if key == "Civitai resources" and isinstance(value, list):
        lines = []
        for resource in value:
            if not isinstance(resource, dict):
                continue
            name = " ".join(str(resource[k]) for k in ("modelName", "modelVersionName") if resource.get(k))
            name = name or resource.get("air") or f"version {resource.get('modelVersionId', '?')}"
            details = ", ".join(f"{k} {resource[k]}" for k in ("type", "weight") if k in resource)
            lines.append(f"{name} ({details})" if details else name)
        return "\n".join(lines)
    if key == "Hashes" and isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def display_rows(metadata: ImageMetadata) -> list[tuple[str, str]]:
    """Label/value pairs for showing the metadata in the UI, most useful first."""
    rows: list[tuple[str, str]] = []
    if metadata.positive:
        rows.append(("Positive", metadata.positive))
    if metadata.negative:
        rows.append(("Negative", metadata.negative))
    for key, value in metadata.settings.items():
        if value not in ("", None, [], {}):
            rows.append((key, _format_value(key, value)))
    rows.append(("Image size", f"{metadata.width}x{metadata.height}"))
    rows.append(("Workflow embedded", "yes" if metadata.has_workflow else "no"))
    return rows
