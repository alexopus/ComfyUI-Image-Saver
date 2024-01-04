import os
import hashlib
from typing import AnyStr
import folder_paths

"""
Given the file path, finds a matching sha256 file, or creates one
based on the headers in the source file
"""
def get_sha256(file_path: AnyStr):
    file_no_ext = os.path.splitext(file_path)[0]
    hash_file = file_no_ext + ".sha256"

    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                return f.read().strip()
        except OSError as e:
            print(f"comfy-image-saver: Error reading existing hash file: {e}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    try:
        with open(hash_file, "w") as f:
            f.write(sha256_hash.hexdigest())
    except OSError as e:
        print(f"comfy-image-saver: Error writing hash to {hash_file}: {e}")

    return sha256_hash.hexdigest()

"""
Represent the given embedding name as key as detected by civitAI
"""
def civitai_embedding_key_name(embedding: AnyStr):
    return f'embed:{embedding}'

"""
Represent the given lora name as key as detected by civitAI
NB: this should also work fine for Lycoris
"""
def civitai_lora_key_name(lora: AnyStr):
    return f'LORA:{lora}'

"""
Based on a embedding name, eg: EasyNegative, finds the path as known in comfy, including extension
"""
def full_embedding_path_for(embedding: AnyStr):
    matching_embedding = next((x for x in __list_embeddings() if x.startswith(embedding)), None)
    if matching_embedding == None:
        return None
    return folder_paths.get_full_path("embeddings", matching_embedding)

"""
Based on a lora name, eg: epi_noise_offset2, finds the path as known in comfy, including extension
"""
def full_lora_path_for(lora: AnyStr):
    matching_lora = next((x for x in __list_loras() if x.startswith(lora)), None)
    if matching_lora == None:
        return None
    return folder_paths.get_full_path("loras", matching_lora)

def __list_loras():
    return folder_paths.get_filename_list("loras")

def __list_embeddings():
    return folder_paths.get_filename_list("embeddings")