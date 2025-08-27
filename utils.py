import hashlib
import os
import requests
from typing import Optional, Any
from collections.abc import Collection, Iterator
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

def get_sha256(file_path: str) -> str:
    """
    Given the file path, finds a matching sha256 file, or creates one
    based on the headers in the source file
    """
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

def full_embedding_path_for(embedding: str) -> Optional[str]:
    """
    Based on a embedding name, eg: EasyNegative, finds the path as known in comfy, including extension
    """
    matching_embedding = get_file_path_match("embeddings", embedding)
    if matching_embedding is None:
        print(f'ComfyUI-Image-Saver: could not find full path to embedding "{embedding}"')
        return None
    return folder_paths.get_full_path("embeddings", matching_embedding)

def full_lora_path_for(lora: str) -> Optional[str]:
    """
    Based on a lora name, e.g., 'epi_noise_offset2', finds the path as known in comfy, including extension.
    """
    # Find the matching lora path
    matching_lora = get_file_path_match("loras", lora)
    if matching_lora is None:
        print(f'ComfyUI-Image-Saver: could not find full path to lora "{lora}"')
        return None
    return folder_paths.get_full_path("loras", matching_lora)

def full_checkpoint_path_for(model_name: str) -> str:
    if not model_name:
        return ''

    supported_extensions = set(folder_paths.supported_pt_extensions) | {".gguf"}

    matching_checkpoint = get_file_path_match("checkpoints", model_name, supported_extensions)
    if matching_checkpoint is not None:
        return folder_paths.get_full_path("checkpoints", matching_checkpoint)

    matching_model = get_file_path_match("diffusion_models", model_name, supported_extensions)
    if matching_model:
        return folder_paths.get_full_path("diffusion_models", matching_model)

    print(f'Could not find full path to checkpoint "{model_name}"')
    return ''

def get_file_path_iterator(folder_name: str, supported_extensions: Optional[Collection[str]] = None) -> Iterator[Path]:
    """
    Returns an iterator over valid file paths for the specified model folder.
    """
    if supported_extensions is None:
        return (Path(x) for x in folder_paths.get_filename_list(folder_name))
    else:
        return custom_file_path_generator(folder_name, supported_extensions)

def custom_file_path_generator(folder_name: str, supported_extensions: Collection[str]) -> Iterator[Path]:
    """
    Generator function for file paths, allowing for a customized extension check.
    """
    model_paths = folder_paths.folder_names_and_paths.get(folder_name, [[], set()])[0]
    for path in model_paths:
        if os.path.exists(path):
            base_path = Path(path)
            for root, _, files in os.walk(path):
                root_path = Path(root).relative_to(base_path)
                for file in files:
                    file_path = root_path / file
                    if file_path.suffix.lower() in supported_extensions:
                        yield file_path

def get_file_path_match(folder_name: str, file_name: str, supported_extensions: Optional[Collection[str]] = None) -> Optional[str]:
    supported_extensions_fallback = supported_extensions if supported_extensions is not None else folder_paths.supported_pt_extensions
    file_path = Path(file_name)

    # first try full path match, then fallback to just name match, matching the extension if appropriate
    if file_path.suffix.lower() not in supported_extensions_fallback:
        matching_file_path = next((p for p in get_file_path_iterator(folder_name, supported_extensions) if p.with_suffix('') == file_path), None)
        matching_file_path = (matching_file_path if matching_file_path is not None else
            next((p for p in get_file_path_iterator(folder_name, supported_extensions) if p.stem == file_path.name), None))
    else:
        matching_file_path = next((p for p in get_file_path_iterator(folder_name, supported_extensions) if p == file_path), None)
        matching_file_path = (matching_file_path if matching_file_path is not None else
            next((p for p in get_file_path_iterator(folder_name, supported_extensions) if p.name == file_path.name), None))

    return str(matching_file_path) if matching_file_path is not None else None

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
