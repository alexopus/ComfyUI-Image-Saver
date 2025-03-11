import requests

class CivitaiHashFetcher:
    """
    A ComfyUI custom node that fetches the AutoV3 hash of a model from Civitai
    based on the provided username and model name.
    """

    def __init__(self):
        self.cached_username = None
        self.cached_model_name = None
        self.cached_hash = None  # Stores last fetched hash

    RETURN_TYPES = ("STRING",)  # The node outputs a string (AutoV3 hash)
    FUNCTION = "get_autov3_hash"
    CATEGORY = "CivitaiAPI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "username": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def get_autov3_hash(self, username, model_name):
        """
        Fetches the latest model version from Civitai and extracts its AutoV3 hash.
        Uses caching to avoid redundant API calls.
        """
        # Check if inputs are the same as last time
        if username == self.cached_username and model_name == self.cached_model_name:
            return (self.cached_hash,)  # Return cached value

        base_url = "https://civitai.com/api/v1/models"
        params = {
            "username": username,
            "query": model_name,
            "limit": 1  # Get the most relevant model
        }

        try:
            # Fetch models by username and model name
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                return (f"Error: API request failed with status {response.status_code}",)

            data = response.json()
            items = data.get("items", [])
            if not items:
                return (f"No models found for user '{username}' with name '{model_name}'",)

            # Take the first model from the search results
            model = items[0]
            model_versions = model.get("modelVersions", [])
            if not model_versions:
                return ("No model versions found.",)

            # Assume the first version is the latest
            latest_version = model_versions[0]
            version_id = latest_version.get("id")

            # Fetch detailed version info
            version_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
            version_response = requests.get(version_url)
            if version_response.status_code != 200:
                return (f"Error: Version API request failed with status {version_response.status_code}",)

            version_data = version_response.json()

            # Extract the AutoV3 hash from the model version files
            for file_info in version_data.get("files", []):
                autov3_hash = file_info.get("hashes", {}).get("AutoV3")
                if autov3_hash:
                    # Cache the result before returning
                    self.cached_username = username
                    self.cached_model_name = model_name
                    self.cached_hash = autov3_hash
                    return (autov3_hash,)  # Return the first found hash

            return ("No AutoV3 hash found in version files.",)

        except Exception as e:
            return (f"Error: {e}",)
