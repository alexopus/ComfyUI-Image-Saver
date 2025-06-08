import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import re

from PIL import Image
import torch

import folder_paths
from nodes import MAX_RESOLUTION

from .saver.saver import save_image
from .utils import get_sha256, full_checkpoint_path_for
from .utils_civitai import get_civitai_sampler_name, get_civitai_metadata, MAX_HASH_LENGTH
from .prompt_metadata_extractor import PromptMetadataExtractor

def parse_checkpoint_name(ckpt_name: str) -> str:
    return os.path.basename(ckpt_name)

def parse_checkpoint_name_without_extension(ckpt_name: str) -> str:
    return os.path.splitext(parse_checkpoint_name(ckpt_name))[0]

def get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def save_json(image_info: dict[str, Any], filename: str) -> None:
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')

def make_pathname(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler: str, denoise: float, clip_skip: int) -> str:
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", parse_checkpoint_name(modelname))
    filename = filename.replace("%width", str(width))
    filename = filename.replace("%height", str(height))
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    filename = filename.replace("%sampler_name", sampler_name)
    filename = filename.replace("%steps", str(steps))
    filename = filename.replace("%cfg", str(cfg))
    filename = filename.replace("%scheduler", scheduler)
    filename = filename.replace("%basemodelname", parse_checkpoint_name_without_extension(modelname))
    filename = filename.replace("%denoise", str(denoise))
    filename = filename.replace("%clip_skip", str(clip_skip))
    return filename

def make_filename(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler: str, denoise: float, clip_skip: int) -> str:
    filename = make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)
    return get_timestamp(time_format) if filename == "" else filename

@dataclass
class Metadata:
    modelname: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    clip_skip: int
    additional_hashes: str

class ImageSaverMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "positive":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "negative prompt"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler":             ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
            },
        }

    RETURN_TYPES = ("METADATA",)
    RETURN_NAMES = ("metadata",)
    OUTPUT_TOOLTIPS = ("metadata for Image Saver Simple",)
    FUNCTION = "get_metadata"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Prepare metadata for Image Saver Simple"

    def get_metadata(
        self,
        modelname: str,
        positive: str,
        negative: str,
        width: int,
        height: int,
        seed_value: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        clip_skip: int,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        easy_remix: bool = True,
    ):
        modelname, additional_hashes = ImageSaver.get_multiple_models(modelname, additional_hashes)
        metadata = Metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler, denoise, clip_skip, additional_hashes)
        return (metadata,)

class ImageSaverSimple:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",    {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",   {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",   {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                         "tooltip": "file extension/type to save image as"}),
                "lossless_webp":         ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",      {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "embed_workflow":        ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
                "save_workflow_as_json": ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
            },
            "optional": {
                "metadata":              ("METADATA", {"default": None,                                             "tooltip": "metadata to embed in the image"}),
                "counter":               ("INT",      {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "time_format":           ("STRING",   {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

class ImageSaver:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",   {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",  {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",  {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                        "tooltip": "file extension/type to save image as"}),
            },
            "optional": {
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler":             ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "positive":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "negative prompt"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "lossless_webp":         ("BOOLEAN", {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",     {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN", {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "counter":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "time_format":           ("STRING",  {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
                "embed_workflow":        ("BOOLEAN", {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    # Match 'anything' or 'anything:anything' with trimmed white space
    re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
    # Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
    re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')

    @staticmethod
    def get_multiple_models(modelname: str, additional_hashes: str) -> tuple[str, str]:
        model_names = [m.strip() for m in modelname.split(',')]
        modelname = model_names[0] # Use the first model as the primary one

        # Process additional model names and add to additional_hashes
        for additional_model in model_names[1:]:
            additional_ckpt_path = full_checkpoint_path_for(additional_model)
            if additional_ckpt_path:
                additional_modelhash = get_sha256(additional_ckpt_path)[:10]
                # Add to additional_hashes in "name:HASH" format
                if additional_hashes:
                    additional_hashes += ","
                additional_hashes += f"{additional_model}:{additional_modelhash}"
        return modelname, additional_hashes

    def save_files(
        self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        steps: int,
        cfg: float,
        modelname: str,
        sampler_name: str,
        scheduler: str,
        positive: str,
        negative: str,
        seed_value: int,
        width: int,
        height: int,
        lossless_webp: bool,
        quality_jpeg_or_webp: int,
        optimize_png: bool,
        counter: int,
        denoise: float,
        clip_skip: int,
        time_format: str,
        save_workflow_as_json: bool = False,
        embed_workflow: bool = True,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        easy_remix: bool = True,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modelname, additional_hashes = ImageSaver.get_multiple_models(modelname, additional_hashes)
        filename = make_filename(filename, width, height, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)
        path = make_pathname(path, width, height, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)

        ckpt_path = full_checkpoint_path_for(modelname)
        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler)
        basemodelname = parse_checkpoint_name_without_extension(modelname)

        # Get existing hashes from model, loras, and embeddings
        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
        # Parse manual hashes
        manual_entries = ImageSaver.parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
        # Get Civitai metadata
        civitai_resources, hashes, add_model_hash = get_civitai_metadata(modelname, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data)

        if easy_remix:
            positive = ImageSaver.clean_prompt(positive, metadata_extractor)
            negative = ImageSaver.clean_prompt(negative, metadata_extractor)

        positive_a111_params = positive.strip()
        negative_a111_params = f"\nNegative prompt: {negative.strip()}"
        clip_skip_str = f", Clip skip: {abs(clip_skip)}" if clip_skip != 0 else ""
        model_hash_str = f", Model hash: {add_model_hash}" if add_model_hash else ""
        hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

        a111_params = (
            f"{positive_a111_params}{negative_a111_params}\n"
            f"Steps: {steps}, Sampler: {civitai_sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, "
            f"Size: {width}x{height}{clip_skip_str}{model_hash_str}, Model: {basemodelname}{hashes_str}, Version: ComfyUI"
        )

        # Add Civitai resource listing
        if download_civitai_data and civitai_resources:
            a111_params += f", Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}"

        output_path = os.path.join(folder_paths.output_directory, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        filenames = ImageSaver.save_images(images, output_path, filename, a111_params, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow)

        subfolder = os.path.normpath(path)

        final_hashes = ",".join(f"{Path(name.split(':')[-1]).stem + ':' if name else ''}{hash}{':' + str(weight) if weight is not None and download_civitai_data else ''}" for name, (_, weight, hash) in ({ modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries).items())

        return {
            "result": (final_hashes, a111_params),
            "ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)},
        }

    @staticmethod
    def save_images(images: list[torch.Tensor], output_path: str, filename_prefix: str, a111_params: str, extension: str, quality_jpeg_or_webp: int, lossless_webp: bool, optimize_png: bool, prompt: dict[str, Any] | None, extra_pnginfo: dict[str, Any] | None, save_workflow_as_json: bool, embed_workflow: bool) -> list[str]:
        paths: list[str] = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            current_filename_prefix = ImageSaver.get_unique_filename(output_path, filename_prefix, extension)
            filename = f"{current_filename_prefix}.{extension}"
            filepath = os.path.join(output_path, filename)

            save_image(img, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, a111_params, prompt, extra_pnginfo, embed_workflow)

            if save_workflow_as_json:
                save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

            paths.append(filename)
        return paths

    @staticmethod
    def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
        """Process additional_hashes input (a string) by normalizing, removing extra spaces/newlines, and splitting by comma"""
        manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
        unnamed_count = 0

        additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
        for entry in additional_hash_split:
            match = (ImageSaver.re_manual_hash_weights if download_civitai_data else ImageSaver.re_manual_hash).search(entry)
            if match is None:
                print(f"ComfyUI-Image-Saver: Invalid additional hash string: '{entry}'")
                continue

            groups = tuple(group for group in match.groups() if group)

            # Read weight and remove from groups, if needed
            weight = None
            if download_civitai_data and len(groups) > 1:
                try:
                    weight = float(groups[-1])
                    groups = groups[:-1]
                except (ValueError, TypeError):
                    pass

            # Read hash, optionally preceded by name
            name, hash = groups if len(groups) > 1 else (None, groups[0])

            if len(hash) > MAX_HASH_LENGTH:
                print(f"ComfyUI-Image-Saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
                continue

            if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
                print(f"ComfyUI-Image-Saver: Skipping duplicate hash: {hash}")
                continue  # Skip duplicates

            if hash.lower() in existing_hashes:
                print(f"ComfyUI-Image-Saver: Skipping manual hash already present in resources: {hash}")
                continue

            if name is None:
                unnamed_count += 1
                name = f"manual{unnamed_count}"
            elif name in manual_entries:
                print(f"ComfyUI-Image-Saver: Duplicate manual hash name '{name}' is being overwritten.")

            manual_entries[name] = (None, weight, hash)

            if len(manual_entries) > 29:
                print("ComfyUI-Image-Saver: Reached maximum limit of 30 manual hashes. Skipping the rest.")
                break

        return manual_entries

    @staticmethod
    def clean_prompt(prompt: str, metadata_extractor: PromptMetadataExtractor) -> str:
        """Clean prompts for easier remixing by removing LoRAs and simplifying embeddings."""
        # Strip loras
        prompt = re.sub(metadata_extractor.LORA, "", prompt)
        # Shorten 'embedding:path/to/my_embedding' -> 'my_embedding'
        # Note: Possible inaccurate embedding name if the filename has been renamed from the default
        prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
        # Remove prompt control edits. e.g., 'STYLE(A1111, mean)', 'SHIFT(1)`, etc.`
        prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
        return prompt

    @staticmethod
    def get_unique_filename(output_path: str, filename_prefix: str, extension: str) -> str:
        existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]

        if not existing_files:
            return f"{filename_prefix}"

        suffixes = []
        for f in existing_files:
            name, _ = os.path.splitext(f)
            parts = name.split('_')
            if parts[-1].isdigit():
                suffixes.append(int(parts[-1]))

        if suffixes:
            next_suffix = max(suffixes) + 1
        else:
            next_suffix = 1

        return f"{filename_prefix}_{next_suffix:02d}"
