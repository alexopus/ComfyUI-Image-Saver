import re
from typing import List, AnyStr

from .utils import civitai_embedding_key_name, civitai_lora_key_name, full_embedding_path_for, full_lora_path_for, get_sha256

"""
Extracts Embeddings and Lora's from the given prompts
and allows asking for their sha's 
"""
class PromptMetadataExtractor:
    # Anything that follows embedding:<characters except , or whitespace
    EMBEDDING = r'embedding:([^,\s\(\)\:]+)'
    # Anything that follows <lora:NAME> with allowance for :weight, :weight.fractal or LBW
    LORA = r'<lora:([^>:]+)(?::.+)?>'

    def __init__(self, prompts: List[AnyStr]):
        self.__embeddings = {}
        self.__loras = {}
        self.__perform(prompts)

    """
    Returns the embeddings used in the given prompts in a format as known by civitAI
    """
    def get_embeddings(self):
        return self.__embeddings
        
    """
    Returns the lora's used in the given prompts in a format as known by civitAI
    """
    def get_loras(self):
        return self.__loras

    # Private API
    def __perform(self, prompts):
        for prompt in prompts:
            embeddings = re.findall(self.EMBEDDING, prompt, re.IGNORECASE | re.MULTILINE)
            
            for embedding in embeddings:
                self.__extract_embedding_information(embedding)
            
            loras = re.findall(self.LORA, prompt, re.IGNORECASE | re.MULTILINE)
            for lora in loras:
                self.__extract_lora_information(lora)

    def __extract_embedding_information(self, embedding: AnyStr):
        embedding_name = civitai_embedding_key_name(embedding)
        embedding_path = full_embedding_path_for(embedding)
        if embedding_path == None:
            return
        sha = self.__get_shortened_sha(embedding_path)
        self.__embeddings[embedding_name] = sha

    def __extract_lora_information(self, lora: AnyStr):
        lora_name = civitai_lora_key_name(lora)
        lora_path = full_lora_path_for(lora)
        if lora_path == None:
            return
        sha = self.__get_shortened_sha(lora_path)
        self.__loras[lora_name] = sha
    
    def __get_shortened_sha(self, file_path: AnyStr):
       return get_sha256(file_path)[:10]
