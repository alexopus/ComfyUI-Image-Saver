import os
from datetime import datetime
from pathlib import Path
import json
import requests
import piexif
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import re
import folder_paths
from .utils import get_sha256
from .prompt_metadata_extractor import PromptMetadataExtractor
from nodes import MAX_RESOLUTION

def parse_checkpoint_name(ckpt_name):
    return os.path.basename(ckpt_name)

def parse_checkpoint_name_without_extension(ckpt_name):
    return os.path.splitext(parse_checkpoint_name(ckpt_name))[0]

def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def save_json(image_info, filename):
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')

def make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip):
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

def make_filename(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip):
    filename = make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)
    return get_timestamp(time_format) if filename == "" else filename

class ImageSaver:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.civitai_sampler_map = {
            'euler_ancestral': 'Euler a',
            'euler': 'Euler',
            'lms': 'LMS',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_3m_sde': 'DPM++ 3M SDE',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc_bh2': 'UniPC',
            'uni_pc': 'UniPC',
            'lcm': 'LCM',
        }

    def get_civitai_sampler_name(self, sampler_name, scheduler):
        # based on: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts#L122
        if sampler_name in self.civitai_sampler_map:
            civitai_name = self.civitai_sampler_map[sampler_name]

            if scheduler == "karras":
                civitai_name += " Karras"
            elif scheduler == "exponential":
                civitai_name += " Exponential"

            return civitai_name
        else:
            if scheduler != 'normal':
                return f"{sampler_name}_{scheduler}"
            else:
                return sampler_name

    @classmethod
    def INPUT_TYPES(cls):
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
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name"}),
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
                "embed_workflow_in_png": ("BOOLEAN", {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved PNG files"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    # Match 'anything' or 'anything:anything' with trimmed white space
    re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
    # Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
    re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')
    MAX_HASH_LENGTH = 16 # skip larger unshortened hashes, such as full sha256 or blake3

    def save_files(
        self,
        images,
        seed_value,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        modelname,
        quality_jpeg_or_webp,
        lossless_webp,
        optimize_png,
        width,
        height,
        counter,
        filename,
        path,
        extension,
        time_format,
        denoise,
        clip_skip,
        additional_hashes="",
        save_workflow_as_json=False,
        embed_workflow_in_png=True,
        download_civitai_data=False,
        easy_remix=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        filename = make_filename(filename, width, height, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)
        path = make_pathname(path, width, height, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise, clip_skip)
        ckpt_path = folder_paths.get_full_path("checkpoints", modelname)

        if not ckpt_path:
            ckpt_path = folder_paths.get_full_path("diffusion_models", modelname)

        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = self.get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler)

        # Process additional_hashes input (a string) by normalizing, removing extra spaces/newlines, and splitting by comma
        manual_entries = {}
        unnamed_count = 0
        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}  # Get set of parsed hashes
        additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
        for entry in additional_hash_split:
            match = (self.re_manual_hash_weights if download_civitai_data else self.re_manual_hash).search(entry)
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

            if len(hash) > self.MAX_HASH_LENGTH:
                print(f"ComfyUI-Image-Saver: Skipping hash. Length exceeds maximum of {self.MAX_HASH_LENGTH} characters: {hash}")
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

        # Download or load cache of Civitai data, save specially-formatted data to metadata
        civitai_resources = []
        hashes = {}
        add_model_hash = None
        if download_civitai_data:
            for name, (filepath, weight, hash) in ({ modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries).items():
                civitai_info = self.get_civitai_info(filepath, hash)
                if civitai_info is not None:
                    resource_data = {}

                    # Optional data - modelName, versionName
                    resource_data["modelName"] = civitai_info["model"]["name"]
                    resource_data["versionName"] = civitai_info["name"]
                    
                    # Weight/strength (for LoRA or embedding)
                    if weight is not None:
                        resource_data["weight"] = weight
                    
                    # Required data - AIR or modelVersionId (unique resource identifier)
                    # https://github.com/civitai/civitai/wiki/AIR-%E2%80%90-Uniform-Resource-Names-for-AI
                    if "air" in civitai_info:
                        resource_data["air"] = civitai_info["air"]
                    else:
                        # Fallback if AIR is not found
                        resource_data["modelVersionId"] = civitai_info["id"]
                    civitai_resources.append(resource_data)
                else:
                    # Fallback in case the data wasn't loaded to add to the "Hashes" section
                    if name == modelname:
                        add_model_hash = hash.upper()
                    else:
                        hashes[name] = hash.upper()
        else:
            # Convert all hashes to JSON format
            hashes = {key: value[2] for key, value in embeddings.items()} | {key: value[2] for key, value in loras.items()} | {key: value[2] for key, value in manual_entries.items()} | {"model": modelhash}
            add_model_hash = modelhash
        basemodelname = parse_checkpoint_name_without_extension(modelname)

        if easy_remix:
            def clean_prompt(prompt: str) -> str:
                # Strip loras
                prompt = re.sub(metadata_extractor.LORA, "", prompt)
                # Shorten 'embedding:path/to/my_embedding' -> 'my_embedding'
                # Note: Possible inaccurate embedding name if the filename has been renamed from the default
                prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
                # Remove prompt control edits. e.g., 'STYLE(A1111, mean)', 'SHIFT(1)`, etc.`
                prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
                return prompt

            positive = clean_prompt(positive)
            negative = clean_prompt(negative)

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

        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        filenames = self.save_images(images, output_path, filename, a111_params, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow_in_png)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, a111_params, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow_in_png) -> list[str]:
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            current_filename_prefix = self.get_unique_filename(output_path, filename_prefix, extension)
            filename = f"{current_filename_prefix}.{extension}"
            filepath = os.path.join(output_path, filename)

            if extension == 'png':
                metadata = PngInfo()
                metadata.add_text("parameters", a111_params)

                if embed_workflow_in_png:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                img.save(filepath, pnginfo=metadata, optimize=optimize_png)
            else: # webp & jpeg
                img.save(filepath, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(a111_params, encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, filepath)

            if save_workflow_as_json:
                save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

            paths.append(filename)
        return paths

    def get_unique_filename(self, output_path, filename_prefix, extension):
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

    @staticmethod
    def get_manual_folder() -> Path:
        return Path(folder_paths.models_dir) / "image-saver"

    @staticmethod
    def http_get_json(url: str) -> dict | None:
        try:
            response = requests.get(
                url,
                stream=True,
                headers={},
                proxies={ "http": None, "https": None },
                timeout=300
            )
        except TimeoutError:
            print(f"ComfyUI-Image-Saver: HTTP GET Request timed out for {url}")
            return None

        if not response.ok:
            print(f"ComfyUI-Image-Saver: HTTP GET Request failed with error code: {response.status_code}: {response.reason}")
            return None

        try:
            return response.json()
        except ValueError as e:
            print(f"ComfyUI-Image-Saver: HTTP Response JSON error: {e}")
        return None

    @staticmethod
    def get_civitai_info(path: Path | str | None, model_hash: str) -> dict | None:
        try:
            if not model_hash:
                print("ComfyUI-Image-Saver: Error: Missing hash.")
                return None

            # path is None for additional hashes added by the user - caches manually added hash data in the "image-saver" folder
            if path is None:
                manual_list = ImageSaver.get_manual_list()
                manual_data = manual_list.get(model_hash.upper(), None)
                if manual_data is None:
                    content = ImageSaver.download_model_info(path, model_hash)
                    if content is None:
                        return None
                    
                    # dynamically receive filename from the website to save the metadata
                    file = next((file for file in content["files"] if any(len(value) <= ImageSaver.MAX_HASH_LENGTH and value.upper() == model_hash.upper() for value in file["hashes"].values())), None)
                    if file is None:
                        print(f"ComfyUI-Image-Saver: ({model_hash}) No file hash matched in metadata (should be impossible)")
                        return content
                    filename = file["name"]

                    # Cache data in a local file, removing the need for repeat http requests
                    for hash_value in file["hashes"].values():
                        if len(hash_value) <= ImageSaver.MAX_HASH_LENGTH:
                            manual_list = ImageSaver.append_manual_list(hash_value.upper(), { "filename": filename, "type": content["model"]["type"] })

                    ImageSaver.save_civitai_info_file(content, ImageSaver.get_manual_folder() / filename)
                    return content
                else:
                    path = ImageSaver.get_manual_folder() / manual_data["filename"]

            info_path = Path(path).with_suffix(".civitai.info").absolute()
            with open(info_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return ImageSaver.download_model_info(path, model_hash)
        except Exception as e:
            print(f"ComfyUI-Image-Saver: Civitai info error: {e}")
        return None

    @staticmethod
    def get_manual_list() -> dict[str, dict]:
        folder = ImageSaver.get_manual_folder()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            manual_path = (folder / "manual-hashes.json").absolute()
            with open(manual_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"ComfyUI-Image-Saver: Manual list get error: {e}")
        return {}

    @staticmethod
    def append_manual_list(key: str, value: dict) -> dict[str, dict]:
        manual_list = ImageSaver.get_manual_list() | { key: value }
        try:
            with open((ImageSaver.get_manual_folder() / "manual-hashes.json").absolute(), 'w') as file:
                file.write(json.dumps(manual_list, indent=4))
        except Exception as e:
            print(f"ComfyUI-Image-Saver: Manual list append error: {e}")
        return manual_list

    @staticmethod
    def download_model_info(path: Path | str | None, model_hash: str) -> dict | None:
        model_label = model_hash if path is None else f"{Path(path).stem}:{model_hash}"
        print(f"ComfyUI-Image-Saver: Downloading model info for '{model_label}'.")

        content = ImageSaver.http_get_json(f'https://civitai.com/api/v1/model-versions/by-hash/{model_hash.upper()}')
        if content is None:
            return None
        model_id = content["modelId"]
        parent_model = ImageSaver.http_get_json(f'https://civitai.com/api/v1/models/{model_id}')
        if not parent_model:
            parent_model = {}

        content["creator"] = parent_model.get("creator", "{}")
        model_metadata = content["model"]
        for metadata in [ "description", "tags", "allowNoCredit", "allowCommercialUse", "allowDerivatives", "allowDifferentLicense" ]:
            model_metadata[metadata] = parent_model.get(metadata, "")

        if path is not None:
            ImageSaver.save_civitai_info_file(content, path)
        
        return content

    @staticmethod
    def save_civitai_info_file(content: dict, path: Path | str) -> bool:
        try:
            with open(Path(path).with_suffix(".civitai.info").absolute(), 'w') as info_file:
                info_file.write(json.dumps(content, indent=4))
        except Exception as e:
            print(f"ComfyUI-Image-Saver: Save Civitai info error '{path}': {e}")
            return False
        return True
