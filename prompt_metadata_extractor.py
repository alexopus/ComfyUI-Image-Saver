import re
from typing import List, AnyStr
import folder_paths

from .utils import get_sha256

"""
Extracts Embeddings and Lora's from the given prompts
and allows asking for their sha's 
"""
class PromptMetadataExtractor:
    # Anything that follows embedding:<characters except , or whitespace
    EMBEDDING = r'embedding:([^,\s\(\)\:]+)'
    # Anything that follows <lora:NAME> with allowance for :weight, :weight.fractal or LBW
    LORA = r'<lora:([^:]+)(?::[1-9]+\.?[1-9]?.*)?>'

    def __init__(self, prompts: List[AnyStr]):
        self.prompts = prompts
        self._embeddings = {}
        self._loras = {}
        self._performed = False

    def perform(self):
        for prompt in self.prompts:
            print(prompt)
            embeddings = re.findall(self.EMBEDDING, prompt, re.IGNORECASE | re.MULTILINE)
            print(embeddings)
            
            for embedding in embeddings:
                self.extract_embedding_information(embedding)
            
            loras = re.findall(self.LORA, prompt, re.IGNORECASE | re.MULTILINE)
            for lora in loras:
                self.extract_lora_information(lora)
        self._performed = True

    def get_embeddings(self):
        if not self._performed:
            self.perform()
        return self._embeddings
        
    def get_loras(self):
        if not self._performed:
            self.perform()
        return self._loras

    def extract_embedding_information(self, embedding: AnyStr):
        embedding_name = self.civitai_embedding_key_name(embedding)
        sha = self.get_hash(self.embedding_path(embedding))
        self._embeddings[embedding_name] = sha

    def extract_lora_information(self, lora: AnyStr):
        lora_name = self.civitai_lora_key_name(lora)
        sha = self.get_hash(self.lora_path(lora))
        self._loras[lora_name] = sha
         
    def embedding_path(self, embedding: AnyStr):
        matching_embedding = next(x for x in self.embeddings() if x.startswith(embedding))
        return folder_paths.get_full_path("embeddings", matching_embedding)
    
    def lora_path(self, lora: AnyStr):
        matching_lora = next(x for x in self.loras() if x.startswith(lora))
        return folder_paths.get_full_path("loras", matching_lora)
    
    def loras(self):
        return folder_paths.get_filename_list("loras")

    def embeddings(self):
        return folder_paths.get_filename_list("embeddings")
    
    def get_hash(self, file_path: AnyStr):
       return get_sha256(file_path)[:10]

    def civitai_embedding_key_name(self, embedding: AnyStr):
       return f'embed:{embedding}'

    def civitai_lora_key_name(self, lora: AnyStr):
       return f'LORA:{lora}'
    