import hashlib
import os
import requests
from typing import Optional, Any
from pathlib import Path
from tqdm import tqdm
import folder_paths
import re

def sanitize_filename(filename: str) -> str:
    """Remove characters that are unsafe for filenames."""
    # Remove characters that are generally unsafe across file systems
    unsafe_chars = r'[<>:"|?*\x00-\x1f]'
    sanitized = re.sub(unsafe_chars, '', filename)

    # Remove trailing periods and spaces (problematic on Windows)
    sanitized = sanitized.rstrip('. ')
    return sanitized

"""
Given the file path, finds a matching sha256 file, or creates one
based on the headers in the source file
"""
def get_sha256(file_path: str) -> str:
    file_no_ext = os.path.splitext(file_path)[0]
    hash_file = file_no_ext + ".sha256"

    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                return f.read().strip()
        except OSError as e:
            print(f"ComfyUI-Image-Saver: Error reading existing hash file: {e}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        block_size = 1048576 # 1 MB

        print(f"ComfyUI-Image-Saver: Calculating sha256 for '{Path(file_path).stem}'")
        with tqdm(None, None, file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
            for byte_block in iter(lambda: f.read(block_size), b""):
                progress_bar.update(len(byte_block))
                sha256_hash.update(byte_block)

    try:
        with open(hash_file, "w") as f:
            f.write(sha256_hash.hexdigest())
    except OSError as e:
        print(f"ComfyUI-Image-Saver: Error writing hash to {hash_file}: {e}")

    return sha256_hash.hexdigest()

"""
Based on a embedding name, eg: EasyNegative, finds the path as known in comfy, including extension
"""
def full_embedding_path_for(embedding: str) -> Optional[str]:
    matching_embedding = next((x for x in folder_paths.get_filename_list("embeddings") if x.startswith(embedding)), None)
    if matching_embedding == None:
        return None
    return folder_paths.get_full_path("embeddings", matching_embedding)

"""
Based on a lora name, e.g., 'epi_noise_offset2', finds the path as known in comfy, including extension.
"""
def full_lora_path_for(lora: str) -> Optional[str]:
    last_dot_position = lora.rfind('.')
    extension = lora[last_dot_position:] if last_dot_position != -1 else ""
    if extension not in folder_paths.supported_pt_extensions:
        lora += ".safetensors"

    # Find the matching lora path
    matching_lora = next((x for x in folder_paths.get_filename_list("loras") if x.endswith(lora)), None)
    if matching_lora is None:
        print(f'ComfyUI-Image-Saver: could not find full path to lora "{lora}"')
        return None
    return folder_paths.get_full_path("loras", matching_lora)

def full_checkpoint_path_for(model_name: str) -> str:
    if not model_name:
        return ''

    last_dot_position = model_name.rfind('.')
    extension = model_name[last_dot_position:] if last_dot_position != -1 else ""
    supported_extensions = set(folder_paths.supported_pt_extensions) | {".gguf"}
    if extension.lower() not in supported_extensions:
        model_name += ".safetensors"

    matching_checkpoint = next((x for x in folder_paths.get_filename_list("checkpoints") if x.endswith(model_name)), None)
    if matching_checkpoint:
        return folder_paths.get_full_path("checkpoints", matching_checkpoint)

    matching_model = next((x for x in folder_paths.get_filename_list("diffusion_models") if x.endswith(model_name)), None)
    if matching_model:
        return folder_paths.get_full_path("diffusion_models", matching_model)

    # Handle GGUF files in diffusion_models directories since they're not in supported_pt_extensions
    if model_name.endswith('.gguf'):
        diffusion_model_paths = folder_paths.folder_names_and_paths.get("diffusion_models", [[], set()])[0]
        for path in diffusion_model_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(model_name):
                            return os.path.join(root, file)

    print(f'Could not find full path to checkpoint "{model_name}"')
    return ''

def http_get_json(url: str) ->  dict[str, Any] | None:
    try:
        response = requests.get(url, timeout=300)
    except requests.exceptions.Timeout:
        print(f"ComfyUI-Image-Saver: HTTP GET Request timed out for {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"ComfyUI-Image-Saver: Warning - Network connection error for {url}: {e}")
        return None

    if not response.ok:
        print(f"ComfyUI-Image-Saver: HTTP GET Request failed with error code: {response.status_code}: {response.reason}")
        return None

    try:
        return response.json()
    except ValueError as e:
        print(f"ComfyUI-Image-Saver: HTTP Response JSON error: {e}")
    return None
