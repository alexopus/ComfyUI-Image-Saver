import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import folder_paths

from .utils import http_get_json

MAX_HASH_LENGTH = 16 # skip larger unshortened hashes, such as full sha256 or blake3

"""
Represent the given embedding name as key as detected by civitAI
"""
def civitai_embedding_key_name(embedding: str) -> str:
    return f'embed:{embedding}'

"""
Represent the given lora name as key as detected by civitAI
NB: this should also work fine for Lycoris
"""
def civitai_lora_key_name(lora: str) -> str:
    return f'LORA:{lora}'

CIVITAI_SAMPLER_MAP = {
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

def get_civitai_sampler_name(sampler_name: str, scheduler: str) -> str:
    # based on: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts#L122
    if sampler_name in CIVITAI_SAMPLER_MAP:
        civitai_name = CIVITAI_SAMPLER_MAP[sampler_name]

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

def get_civitai_metadata(
        modelname: str,
        ckpt_path: str,
        modelhash: str,
        loras: Dict[str, Tuple[str, float, str]],
        embeddings: Dict[str, Tuple[str, float, str]],
        manual_entries: Dict[str, tuple[str | None, float | None, str]],
        download_civitai_data: bool) -> Tuple[List[Dict[str, str | float]], Dict[str, str], str | None]:
    """Download or load cache of Civitai data, save specially-formatted data to metadata"""
    civitai_resources: List[Dict[str, str | float]] = []
    hashes = {}
    add_model_hash = None

    if download_civitai_data:
        for name, (filepath, weight, hash) in ({ modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries).items():
            civitai_info = get_civitai_info(filepath, hash)
            if civitai_info is not None:
                resource_data: Dict[str, str | float] = {}

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

    return civitai_resources, hashes, add_model_hash

def get_civitai_info(path: Path | str | None, model_hash: str) -> dict[str, Any] | None:
    try:
        if not model_hash:
            print("ComfyUI-Image-Saver: Error: Missing hash.")
            return None

        # path is None for additional hashes added by the user - caches manually added hash data in the "image-saver" folder
        if path is None:
            manual_list = get_manual_list()
            manual_data = manual_list.get(model_hash.upper(), None)
            if manual_data is None:
                content = download_model_info(path, model_hash)
                if content is None:
                    return None

                # dynamically receive filename from the website to save the metadata
                file = next((file for file in content["files"] if any(len(value) <= MAX_HASH_LENGTH and value.upper() == model_hash.upper() for value in file["hashes"].values())), None)
                if file is None:
                    print(f"ComfyUI-Image-Saver: ({model_hash}) No file hash matched in metadata (should be impossible)")
                    return content
                filename = file["name"]

                # Cache data in a local file, removing the need for repeat http requests
                for hash_value in file["hashes"].values():
                    if len(hash_value) <= MAX_HASH_LENGTH:
                        manual_list = append_manual_list(hash_value.upper(), { "filename": filename, "type": content["model"]["type"] })

                save_civitai_info_file(content, get_manual_folder() / filename)
                return content
            else:
                path = get_manual_folder() / manual_data["filename"]

        info_path = Path(path).with_suffix(".civitai.info").absolute()
        with open(info_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return download_model_info(path, model_hash)
    except Exception as e:
        print(f"ComfyUI-Image-Saver: Civitai info error: {e}")
    return None

def download_model_info(path: Path | str | None, model_hash: str) -> dict[str, object] | None:
    model_label = model_hash if path is None else f"{Path(path).stem}:{model_hash}"
    print(f"ComfyUI-Image-Saver: Downloading model info for '{model_label}'.")

    content = http_get_json(f'https://civitai.com/api/v1/model-versions/by-hash/{model_hash.upper()}')
    if content is None:
        return None
    model_id = content["modelId"]
    parent_model = http_get_json(f'https://civitai.com/api/v1/models/{model_id}')
    if not parent_model:
        parent_model = {}

    content["creator"] = parent_model.get("creator", "{}")
    model_metadata = content["model"]
    for metadata in [ "description", "tags", "allowNoCredit", "allowCommercialUse", "allowDerivatives", "allowDifferentLicense" ]:
        model_metadata[metadata] = parent_model.get(metadata, "")

    if path is not None:
        save_civitai_info_file(content, path)

    return content

def save_civitai_info_file(content: dict[str, object], path: Path | str) -> bool:
    try:
        with open(Path(path).with_suffix(".civitai.info").absolute(), 'w') as info_file:
            info_file.write(json.dumps(content, indent=4))
    except Exception as e:
        print(f"ComfyUI-Image-Saver: Save Civitai info error '{path}': {e}")
        return False
    return True

def get_manual_folder() -> Path:
    return Path(folder_paths.models_dir) / "image-saver"

def get_manual_list() -> dict[str, dict[str, Any]]:
    folder = get_manual_folder()
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

def append_manual_list(key: str, value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manual_list = get_manual_list() | { key: value }
    try:
        with open((get_manual_folder() / "manual-hashes.json").absolute(), 'w') as file:
            file.write(json.dumps(manual_list, indent=4))
    except Exception as e:
        print(f"ComfyUI-Image-Saver: Manual list append error: {e}")
    return manual_list
